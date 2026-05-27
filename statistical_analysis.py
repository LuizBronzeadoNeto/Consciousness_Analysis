"""Statistical analysis: did the new features actually improve AUC?

Companion to ``analysis.py``. Reuses ``analysis.load_data`` and the same
feature extraction so the comparison is faithful. Answers two questions:

  Q1. Is the AUC gain from the augmented 8-feature set real, or within
      the margin of error of the 3-feature baseline?
  Q2. Which of the new features carry statistically significant signal?

Writes CSVs and Markdown summaries to ``stats_results/``. Designed to be
re-runnable: tuned hyperparameters and the extracted feature dataframe are
both cached on disk.

Usage:
    python statistical_analysis.py
    python statistical_analysis.py --quick           # 5 repeats, 500 boots
    python statistical_analysis.py --n-repeats 50    # heavier run
    python statistical_analysis.py --no-cache        # ignore cached artifacts
"""

import argparse
import json
import os
import warnings

import numpy
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform, mannwhitneyu, norm, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

import analysis as A

# Heavy CV loops surface harmless sklearn warnings ("collinear features",
# class-imbalance UndefinedMetric). Mute them; surface real errors via except.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*ill-defined.*")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_REPO_ROOT, "stats_results")
os.makedirs(_OUT_DIR, exist_ok=True)

# Feature set definitions. log_ratio = log(delta) - log(alpha), so the raw
# pre-PR feature is exp(log_ratio). delta_alpha_ratio is reconstructed in
# get_features() rather than recomputed from the spectrogram.
FULL = list(A.FEATURE_COLS)
BASELINE_RAW = ["K", "LZ", "delta_alpha_ratio"]
BASELINE_LOG = ["K", "LZ", "log_ratio"]

# Features added on fix/data_preprocessing. log_ratio is tested separately
# because it's a transform of an old feature rather than genuinely new.
NEW_FEATURES = [
    "log_alpha_var",
    "log_theta",
    "log_beta",
    "log_gamma",
    "spectral_entropy",
]
DROP_CANDIDATES = ["K", "LZ"] + ["log_ratio"] + NEW_FEATURES  # everything in FULL

SLICES = ("overall", "propofol", "sevo", "volunteers")


def _features_cache_paths():
    base = os.path.join(_OUT_DIR, "features_full")
    return base + ".parquet", base + ".csv"


def _load_cached_features():
    pq, csv = _features_cache_paths()
    if os.path.exists(pq):
        try:
            return pd.read_parquet(pq)
        except (ImportError, ValueError):
            pass
    if os.path.exists(csv):
        return pd.read_csv(csv)
    return None


def _save_cached_features(df):
    pq, csv = _features_cache_paths()
    try:
        df.to_parquet(pq, index=False)
        return
    except (ImportError, ValueError):
        pass
    df.to_csv(csv, index=False)


def get_features(no_cache=False):
    """Return (df, y, groups). df has columns from FULL plus delta_alpha_ratio."""
    df = None if no_cache else _load_cached_features()
    if df is None:
        with A.time_block("stats_load_data"):
            _, _, df, _ = A.load_data()
        df["delta_alpha_ratio"] = numpy.exp(df["log_ratio"].values)
        _save_cached_features(df)
    else:
        if "delta_alpha_ratio" not in df.columns:
            df["delta_alpha_ratio"] = numpy.exp(df["log_ratio"].values)
    y = df["label"].values.astype(int)
    groups = df["case"].values
    return df, y, groups


def slice_masks(df):
    return {
        "overall": numpy.ones(len(df), dtype=bool),
        "propofol": (df["compound"] == "pure_propofol").values,
        "sevo": df["compound"].isin(["mixed", "pure_sevo"]).values,
        "volunteers": (df["cohort"] != "OR").values,
    }


def _make_pipeline():
    """Same shape as analysis.tuning_svm's CPU branch."""
    return make_pipeline(
        PowerTransformer(method="yeo-johnson", standardize=True),
        SVC(kernel="rbf", class_weight="balanced", probability=False),
    )


def tune_one_set(X, y, groups, n_iter, seed):
    """RandomizedSearchCV over (C, gamma) on the given feature matrix.

    Mirrors analysis.tuning_svm's CPU branch but exposes n_iter so --quick
    can run with fewer search iterations. The full-strength default
    (n_iter=60) matches analysis.py.
    """
    param_dist = {"svc__C": loguniform(0.1, 1000), "svc__gamma": loguniform(1e-4, 1)}
    search = RandomizedSearchCV(
        _make_pipeline(),
        param_dist,
        n_iter=n_iter,
        cv=GroupKFold(n_splits=5),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=seed,
    ).fit(X, y, groups=groups)
    return search.best_params_, float(search.best_score_)


def build_pipeline(best_params):
    pipe = _make_pipeline()
    pipe.set_params(**best_params)
    return pipe


def _tuned_params_cache_path():
    return os.path.join(_OUT_DIR, "tuned_params.json")


def load_or_tune(df, feature_sets, n_iter, seed, refresh=False):
    """Tune each feature set once; cache results to JSON so reruns skip it."""
    cache_path = _tuned_params_cache_path()
    cache = {}
    if not refresh and os.path.exists(cache_path):
        with open(cache_path) as fh:
            cache = json.load(fh)

    y = df["label"].values.astype(int)
    groups = df["case"].values
    tuned = {}
    for name, cols in feature_sets.items():
        key = f"{name}|{','.join(sorted(cols))}|n_iter={n_iter}"
        if key in cache:
            tuned[name] = cache[key]
            continue
        with A.time_block("stats_tune", feature_set=name, n_features=len(cols)):
            X = df[cols].values
            best_params, best_score = tune_one_set(
                X, y, groups, n_iter=n_iter, seed=seed
            )
        tuned[name] = {
            "params": best_params,
            "inner_cv_auc": best_score,
            "features": cols,
        }
        cache[key] = tuned[name]
        with open(cache_path, "w") as fh:
            json.dump(cache, fh, indent=2)
    return tuned


def shuffled_group_kfold(groups, n_splits, rng):
    """GroupKFold-style splits with a fresh random group ordering per call.

    sklearn's GroupKFold is deterministic by group label, so to get
    independent repeats we permute the unique group order ourselves.
    """
    unique = numpy.unique(groups)
    perm = rng.permutation(unique)
    bins = numpy.array_split(perm, n_splits)
    group_to_bin = {g: i for i, b in enumerate(bins) for g in b}
    bin_of = numpy.fromiter(
        (group_to_bin[g] for g in groups), dtype=int, count=len(groups)
    )
    for i in range(n_splits):
        test_idx = numpy.flatnonzero(bin_of == i)
        train_idx = numpy.flatnonzero(bin_of != i)
        yield train_idx, test_idx


def _safe_auc(y_true, y_score):
    """Returns NaN if y_true is single-class, else ROC-AUC."""
    if len(numpy.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def _one_fold(
    X,
    y,
    base_params,
    tr,
    te,
    fold_idx,
    rep_idx,
    feat_cols,
    mask_te_per_slice,
    want_perm,
    perm_seed,
    n_perm,
):
    """Fit one fold; return AUC rows, OOF scores, optional permutation rows."""
    model = build_pipeline(base_params)
    model.fit(X[tr], y[tr])
    score = model.decision_function(X[te])

    auc_rows = []
    for slice_name, mask in mask_te_per_slice.items():
        if mask.sum() == 0:
            auc = float("nan")
        else:
            auc = _safe_auc(y[te][mask], score[mask])
        auc_rows.append(
            {"repeat": rep_idx, "fold": fold_idx, "slice": slice_name, "auc": auc}
        )

    perm_rows = []
    if want_perm:
        base_auc = _safe_auc(y[te], score)
        if numpy.isfinite(base_auc):
            perm_rng = numpy.random.default_rng(perm_seed)
            for fi, feat_name in enumerate(feat_cols):
                deltas = []
                for _ in range(n_perm):
                    Xte_perm = X[te].copy()
                    perm_rng.shuffle(Xte_perm[:, fi])
                    s_perm = model.decision_function(Xte_perm)
                    deltas.append(base_auc - _safe_auc(y[te], s_perm))
                perm_rows.append(
                    {
                        "repeat": rep_idx,
                        "fold": fold_idx,
                        "feature": feat_name,
                        "delta_auc_mean": float(numpy.mean(deltas)),
                        "delta_auc_std": float(numpy.std(deltas)),
                    }
                )

    return auc_rows, perm_rows, te, score


def run_outer_cv(
    df, feat_cols, base_params, n_repeats, seed, want_perm=False, n_perm=20
):
    """Parallel repeated GroupKFold; returns (auc_df, oof_array, perm_df)."""
    X = df[feat_cols].values
    y = df["label"].values.astype(int)
    groups = df["case"].values
    masks = slice_masks(df)

    rng = numpy.random.default_rng(seed)
    jobs = []
    for rep in range(n_repeats):
        rep_seed = int(rng.integers(2**31 - 1))
        rep_rng = numpy.random.default_rng(rep_seed)
        for fold_idx, (tr, te) in enumerate(shuffled_group_kfold(groups, 5, rep_rng)):
            mask_te = {name: m[te] for name, m in masks.items()}
            perm_seed = rep * 1_000_003 + fold_idx
            jobs.append(
                delayed(_one_fold)(
                    X,
                    y,
                    base_params,
                    tr,
                    te,
                    fold_idx,
                    rep,
                    feat_cols,
                    mask_te,
                    want_perm,
                    perm_seed,
                    n_perm,
                )
            )

    results = Parallel(n_jobs=-1, prefer="processes")(jobs)

    auc_rows = []
    perm_rows = []
    oof = numpy.full((n_repeats, len(y)), numpy.nan)
    for auc_r, perm_r, te, score in results:
        auc_rows.extend(auc_r)
        perm_rows.extend(perm_r)
        rep = auc_r[0]["repeat"]
        oof[rep, te] = score

    auc_df = pd.DataFrame(auc_rows)
    perm_df = pd.DataFrame(perm_rows) if perm_rows else None
    return auc_df, oof, perm_df


def paired_wilcoxon_per_slice(auc_full, auc_other):
    """Pair on (repeat, fold, slice). H1: ΔAUC > 0."""
    merged = auc_full.merge(
        auc_other, on=["repeat", "fold", "slice"], suffixes=("_full", "_other")
    )
    merged["delta"] = merged["auc_full"] - merged["auc_other"]
    rows = []
    for slice_name in SLICES:
        sub = merged[merged["slice"] == slice_name]
        valid = sub["delta"].dropna().values
        if len(valid) < 5 or numpy.all(valid == valid[0]):
            p = float("nan")
        else:
            try:
                p = float(wilcoxon(valid, alternative="greater").pvalue)
            except ValueError:
                p = float("nan")
        rows.append(
            {
                "slice": slice_name,
                "median_delta_auc": float(numpy.median(valid))
                if len(valid)
                else float("nan"),
                "mean_delta_auc": float(numpy.mean(valid))
                if len(valid)
                else float("nan"),
                "n_paired": int(len(valid)),
                "wilcoxon_p": p,
            }
        )
    return pd.DataFrame(rows)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values; NaNs pass through."""
    p = numpy.asarray(pvals, dtype=float)
    out = numpy.full_like(p, numpy.nan)
    finite = numpy.isfinite(p)
    n = int(finite.sum())
    if n == 0:
        return out
    finite_idx = numpy.flatnonzero(finite)
    order = finite_idx[numpy.argsort(p[finite_idx])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (n - rank) * p[idx])
        running_max = max(running_max, adj)
        out[idx] = running_max
    return out


def _bca(boot_thetas, jack_thetas, theta_hat, alpha=0.05):
    boot = numpy.asarray(boot_thetas, dtype=float)
    boot = boot[numpy.isfinite(boot)]
    if len(boot) < 10 or not numpy.isfinite(theta_hat):
        return float("nan"), float("nan")
    frac_lt = float(numpy.mean(boot < theta_hat))
    frac_lt = min(max(frac_lt, 1e-6), 1 - 1e-6)
    z0 = norm.ppf(frac_lt)

    jack = numpy.asarray(jack_thetas, dtype=float)
    jack = jack[numpy.isfinite(jack)]
    if len(jack) < 3:
        a = 0.0
    else:
        jm = jack.mean()
        num = ((jm - jack) ** 3).sum()
        den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
        a = float(num / den) if den != 0 else 0.0

    za_lo = norm.ppf(alpha / 2)
    za_hi = norm.ppf(1 - alpha / 2)
    a1 = norm.cdf(z0 + (z0 + za_lo) / (1 - a * (z0 + za_lo)))
    a2 = norm.cdf(z0 + (z0 + za_hi) / (1 - a * (z0 + za_hi)))
    a1 = min(max(a1, 0.0), 1.0)
    a2 = min(max(a2, 0.0), 1.0)
    return float(numpy.quantile(boot, a1)), float(numpy.quantile(boot, a2))


def bootstrap_delta_auc(oof_full, oof_other, y, groups, masks, n_boot, seed):
    """Cluster bootstrap on unique cases; BCa 95% CI per slice."""
    unique = numpy.unique(groups)
    case_rows = {c: numpy.flatnonzero(groups == c) for c in unique}
    rng = numpy.random.default_rng(seed)

    def stat(rows, mask):
        sel = rows[mask[rows]]
        if len(sel) == 0:
            return float("nan")
        a1 = _safe_auc(y[sel], oof_full[sel])
        a2 = _safe_auc(y[sel], oof_other[sel])
        if not (numpy.isfinite(a1) and numpy.isfinite(a2)):
            return float("nan")
        return a1 - a2

    out = {}
    all_rows = numpy.arange(len(y))
    for slice_name, mask in masks.items():
        theta_hat = stat(all_rows, mask)
        boots = numpy.empty(n_boot)
        for b in range(n_boot):
            sample = rng.choice(unique, size=len(unique), replace=True)
            rows = numpy.concatenate([case_rows[c] for c in sample])
            boots[b] = stat(rows, mask)
        jacks = numpy.empty(len(unique))
        for i, c in enumerate(unique):
            rows = numpy.concatenate([case_rows[cc] for cc in unique if cc != c])
            jacks[i] = stat(rows, mask)
        lo, hi = _bca(boots, jacks, theta_hat)
        out[slice_name] = {"delta_auc_point": theta_hat, "ci_low": lo, "ci_high": hi}
    return out


def cluster_mannwhitney(df, feature):
    """Mann-Whitney U on per-(case, state) feature means.

    Each case contributes at most one observation per state, sidestepping the
    within-patient autocorrelation that would inflate a raw row-level test.
    """
    means = df.groupby(["case", "state"])[feature].mean().reset_index()
    cs = means[means["state"] == "Conscious"][feature].dropna().values
    uc = means[means["state"] == "Unconscious"][feature].dropna().values
    if len(cs) < 2 or len(uc) < 2:
        return float("nan"), float("nan")
    try:
        u, p = mannwhitneyu(cs, uc, alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan")
    return float(u), float(p)


def univariate_auc(df, feature):
    y = df["label"].values.astype(int)
    s = df[feature].values
    if not numpy.isfinite(s).all() or numpy.std(s) == 0:
        return float("nan")
    auc = roc_auc_score(y, s)
    return float(max(auc, 1 - auc))


def _fmt(v, spec="+.4f"):
    if v is None or (isinstance(v, float) and not numpy.isfinite(v)):
        return "—"
    return format(v, spec)


def write_q1_md(q1_df, n_repeats, n_boot, path):
    lines = [
        "# Q1 — Did the new features actually improve AUC?\n",
        f"ΔAUC = FULL − baseline. CV columns aggregate {n_repeats} repeats × 5 "
        "folds of shuffled GroupKFold (paired on (repeat, fold)). The OOF "
        f"point estimate and BCa 95% CI come from a {n_boot}-resample "
        "case-level cluster bootstrap on the averaged out-of-fold scores. "
        "Wilcoxon is one-sided (H1: ΔAUC > 0); Holm correction spans all "
        "(baseline × slice) tests in this table.\n",
    ]
    for baseline in sorted(q1_df["baseline"].unique()):
        sub = q1_df[q1_df["baseline"] == baseline]
        sub = sub.set_index("slice").loc[list(SLICES)].reset_index()
        lines.append(f"\n## FULL vs {baseline}\n")
        lines.append(
            "| slice | median ΔAUC (CV) | mean ΔAUC (CV) | "
            "ΔAUC point (OOF) | 95% CI (BCa) | n folds | Wilcoxon p | Holm q |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sub.itertuples(index=False):
            lines.append(
                f"| {r.slice} | {_fmt(r.median_delta_auc_cv)} | "
                f"{_fmt(r.mean_delta_auc_cv)} | {_fmt(r.delta_auc_point_oof)} | "
                f"[{_fmt(r.ci_low)}, {_fmt(r.ci_high)}] | "
                f"{r.n_paired_folds} | {_fmt(r.wilcoxon_p, '.4g')} | "
                f"{_fmt(r.holm_q, '.4g')} |"
            )
    n_total_folds = n_repeats * 5
    lines += [
        "\n## How to read this\n",
        "- A ΔAUC whose CI excludes 0 *and* whose Holm q < 0.05 is a "
        "reliable improvement on that slice.",
        "- A positive median but a CI that crosses 0 means the gain is "
        "plausibly noise — the headline number is consistent with margin "
        "of error.",
        "- The OOF point estimate and CV-median can disagree slightly. "
        "OOF averages prediction scores across repeats first, then scores; "
        "CV-median scores per fold then takes the median. Both are reported.",
        "\n## Caveats\n",
        "- Hyperparameters are tuned once on the full data per feature set "
        "(common-practice shortcut; absolute AUCs carry a small optimism "
        "bias, but the *comparison* is fair).",
        f"- The {n_total_folds} paired folds share training data, so the "
        "Wilcoxon p is slightly anti-conservative. Treat the BCa CI as the "
        "more conservative reference.",
        "- Sevo and volunteer slices are small per fold; their CIs will be "
        "wide regardless of true effect size.",
    ]
    if n_repeats < 20:
        lines.append(
            f"- This run used only {n_repeats} repeats (likely --quick "
            "mode). Numbers are indicative; rerun without --quick before "
            "reporting upward."
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_q2_md(imp_df, path):
    lines = [
        "# Q2 — Which features carry statistically significant signal?\n",
        "Three independent views of each feature, ordered by drop-one ΔAUC:\n",
        "- **perm imp (mean ΔAUC)**: mean drop in test-fold AUC when the "
        "feature column is shuffled inside the test set (positive = "
        "important). Aggregated across all outer-CV folds × permutation "
        "shuffles. Wilcoxon vs zero, Holm-adjusted across features.",
        "- **drop-one ΔAUC**: paired ΔAUC (FULL − FULL_minus_feature) over "
        "the same outer-CV repeats. Positive means dropping the feature "
        "hurts. Wilcoxon, Holm-adjusted across features.",
        "- **case-level Mann-Whitney p**: feature averaged within each "
        "(case, state) pair, then U-test across cases. Sidesteps the "
        "within-patient correlation that would inflate a raw test. "
        "Holm-adjusted across features.",
        "- **univariate AUC**: feature used as a 1D score directly "
        "(max(AUC, 1-AUC)). Descriptive only.\n",
    ]
    cols = [
        ("feature", None),
        ("univariate_auc", ".4f"),
        ("perm_imp_mean", "+.5f"),
        ("perm_wilcoxon_p", ".4g"),
        ("perm_holm_q", ".4g"),
        ("drop_one_median_delta_auc_overall", "+.5f"),
        ("drop_one_wilcoxon_p_overall", ".4g"),
        ("drop_one_holm_q", ".4g"),
        ("case_mannwhitney_p", ".4g"),
        ("case_mannwhitney_holm_q", ".4g"),
    ]
    header = "| " + " | ".join(c for c, _ in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in imp_df.itertuples(index=False):
        cells = []
        for c, spec in cols:
            v = getattr(r, c, None)
            if c == "feature":
                cells.append(str(v))
            else:
                cells.append(_fmt(v, spec) if spec else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=30,
        help="Number of GroupKFold repeats (default: 30)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=60,
        help="RandomizedSearchCV iterations per feature set "
        "(default: 60, matching analysis.tuning_svm)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap resamples for BCa CIs (default: 2000)",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=20,
        help="Permutation shuffles per feature per fold (default: 20)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shrink everything for a ~minutes smoke run",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached features and tuned params",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.n_repeats = min(args.n_repeats, 5)
        args.n_iter = min(args.n_iter, 10)
        args.n_boot = min(args.n_boot, 500)
        args.n_perm = min(args.n_perm, 5)

    with A.time_block(
        "stats_total",
        n_repeats=args.n_repeats,
        n_iter=args.n_iter,
        n_boot=args.n_boot,
    ):
        df, y, groups = get_features(no_cache=args.no_cache)
        print(
            f"Loaded {len(df)} epochs across {len(numpy.unique(groups))} cases. "
            f"Conscious={int((y == 1).sum())} Unconscious={int((y == 0).sum())}."
        )
        masks = slice_masks(df)

        feature_sets = {
            "FULL": FULL,
            "BASELINE_RAW": BASELINE_RAW,
            "BASELINE_LOG": BASELINE_LOG,
        }
        for f in DROP_CANDIDATES:
            feature_sets[f"FULL_minus_{f}"] = [c for c in FULL if c != f]

        tuned = load_or_tune(
            df, feature_sets, n_iter=args.n_iter, seed=args.seed, refresh=args.no_cache
        )
        print("\nTuned inner-CV ROC-AUC per feature set:")
        for name, info in tuned.items():
            print(
                f"  {name:32s} inner_cv_auc={info['inner_cv_auc']:.4f} "
                f"params={info['params']}"
            )

        auc_dfs = {}
        oof_by_set = {}
        perm_df_full = None
        for name, cols in feature_sets.items():
            want_perm = name == "FULL"
            with A.time_block("stats_outer_cv", feature_set=name):
                auc_df, oof_arr, perm_df = run_outer_cv(
                    df,
                    cols,
                    tuned[name]["params"],
                    n_repeats=args.n_repeats,
                    seed=args.seed,
                    want_perm=want_perm,
                    n_perm=args.n_perm,
                )
            auc_df["feature_set"] = name
            auc_dfs[name] = auc_df
            with numpy.errstate(invalid="ignore"):
                oof_by_set[name] = numpy.nanmean(oof_arr, axis=0)
            if perm_df is not None:
                perm_df_full = perm_df
            overall = auc_df[auc_df["slice"] == "overall"]["auc"].dropna()
            print(
                f"  [{name:32s}] overall AUC median={overall.median():.4f} "
                f"IQR=[{overall.quantile(0.25):.4f}, {overall.quantile(0.75):.4f}]"
            )

        all_auc_df = pd.concat(auc_dfs.values(), ignore_index=True)
        all_auc_df.to_csv(os.path.join(_OUT_DIR, "auc_comparison.csv"), index=False)

        q1_rows = []
        for baseline_name in ["BASELINE_RAW", "BASELINE_LOG"]:
            wilcox = paired_wilcoxon_per_slice(auc_dfs["FULL"], auc_dfs[baseline_name])
            with A.time_block("stats_bootstrap", baseline=baseline_name):
                boot = bootstrap_delta_auc(
                    oof_by_set["FULL"],
                    oof_by_set[baseline_name],
                    y,
                    groups,
                    masks,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
            for r in wilcox.itertuples(index=False):
                bi = boot[r.slice]
                q1_rows.append(
                    {
                        "baseline": baseline_name,
                        "slice": r.slice,
                        "median_delta_auc_cv": r.median_delta_auc,
                        "mean_delta_auc_cv": r.mean_delta_auc,
                        "delta_auc_point_oof": bi["delta_auc_point"],
                        "ci_low": bi["ci_low"],
                        "ci_high": bi["ci_high"],
                        "n_paired_folds": r.n_paired,
                        "wilcoxon_p": r.wilcoxon_p,
                    }
                )
        q1_df = pd.DataFrame(q1_rows)
        q1_df["holm_q"] = holm(q1_df["wilcoxon_p"].values)
        q1_df.to_csv(os.path.join(_OUT_DIR, "auc_comparison_test.csv"), index=False)

        # (a) Permutation importance from the FULL outer-CV loop
        perm_agg = (
            perm_df_full.groupby("feature")["delta_auc_mean"]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "perm_imp_mean",
                    "median": "perm_imp_median",
                    "std": "perm_imp_std",
                    "count": "perm_imp_n",
                }
            )
        )
        perm_pvals = {}
        for feat in perm_agg["feature"]:
            d = perm_df_full[perm_df_full["feature"] == feat]["delta_auc_mean"].values
            if len(d) >= 5 and not numpy.all(d == d[0]):
                try:
                    perm_pvals[feat] = float(wilcoxon(d, alternative="greater").pvalue)
                except ValueError:
                    perm_pvals[feat] = float("nan")
            else:
                perm_pvals[feat] = float("nan")
        perm_agg["perm_wilcoxon_p"] = perm_agg["feature"].map(perm_pvals)

        # (b) Drop-one ΔAUC vs FULL (overall slice — keep the table readable)
        drop_rows = []
        for f in DROP_CANDIDATES:
            wilcox = paired_wilcoxon_per_slice(
                auc_dfs["FULL"], auc_dfs[f"FULL_minus_{f}"]
            )
            ov = wilcox[wilcox["slice"] == "overall"].iloc[0]
            drop_rows.append(
                {
                    "feature": f,
                    "drop_one_median_delta_auc_overall": ov["median_delta_auc"],
                    "drop_one_wilcoxon_p_overall": ov["wilcoxon_p"],
                }
            )
        drop_df = pd.DataFrame(drop_rows)

        # (c) Cluster Mann-Whitney + (d) univariate AUC
        feats_for_table = DROP_CANDIDATES + ["delta_alpha_ratio"]
        mw_rows = []
        for f in feats_for_table:
            u, p = cluster_mannwhitney(df, f)
            mw_rows.append(
                {
                    "feature": f,
                    "case_mannwhitney_u": u,
                    "case_mannwhitney_p": p,
                    "univariate_auc": univariate_auc(df, f),
                }
            )
        mw_df = pd.DataFrame(mw_rows)

        imp = mw_df.merge(drop_df, on="feature", how="left").merge(
            perm_agg, on="feature", how="left"
        )
        imp["perm_holm_q"] = holm(imp["perm_wilcoxon_p"].values)
        imp["drop_one_holm_q"] = holm(imp["drop_one_wilcoxon_p_overall"].values)
        imp["case_mannwhitney_holm_q"] = holm(imp["case_mannwhitney_p"].values)
        imp = imp.sort_values(
            "drop_one_median_delta_auc_overall", ascending=False, na_position="last"
        ).reset_index(drop=True)
        imp.to_csv(os.path.join(_OUT_DIR, "feature_importance.csv"), index=False)

        write_q1_md(
            q1_df,
            args.n_repeats,
            args.n_boot,
            os.path.join(_OUT_DIR, "auc_comparison_summary.md"),
        )
        write_q2_md(imp, os.path.join(_OUT_DIR, "feature_importance_summary.md"))

        print(f"\nWrote results to {_OUT_DIR}:")
        for fname in [
            "auc_comparison.csv",
            "auc_comparison_test.csv",
            "auc_comparison_summary.md",
            "feature_importance.csv",
            "feature_importance_summary.md",
        ]:
            print(f"  - {fname}")


if __name__ == "__main__":
    main()
