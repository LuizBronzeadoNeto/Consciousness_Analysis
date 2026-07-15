import numpy
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GroupKFold
import complexity_calculations as eeg
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import loguniform, gaussian_kde, norm
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from sklearn.model_selection import RandomizedSearchCV
from joblib import Parallel, delayed
import argparse
import logging
import json
import time
import contextlib
import datetime
import sys
import yaml

OUTLIER_THRESHOLD = 35
WINDOW_BINS = 30
WINDOW_STRIDE = 15
QUALITY_MIN_FRAC = 0.8
FEATURE_COLS = [
    "K",
    "LZ_Classic",
    "POLZ_4",
    "POLZ_8",
    "POLZ_16",
    "POLZ_32", # <-- NOVA LINHA
    "log_ratio",
    "log_alpha_var",
    "log_theta",
    "log_beta",
    "log_gamma",
    "spectral_entropy",
]
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_PERF_DIR = os.path.join(_REPO_ROOT, "perf_logs")
os.makedirs(_PERF_DIR, exist_ok=True)
_PERF_PATH = os.path.join(_PERF_DIR, f"perf_{_RUN_ID}.log")
_CACHE_DIR = os.path.join(_REPO_ROOT, ".cache")
_CACHE_DISABLED = os.environ.get("CONS_NO_CACHE", "0") == "1"

logger = logging.getLogger("cons_analysis")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _stream = logging.StreamHandler(sys.stderr)
    _stream.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(_stream)
    logger.propagate = False


def _emit_perf(record):
    with open(_PERF_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _make_record(name, elapsed_s, fields):
    return {
        "ts": time.time(),
        "run_id": _RUN_ID,
        "block": name,
        "elapsed_s": round(elapsed_s, 6),
        **fields,
    }


@contextlib.contextmanager
def time_block(name, **fields):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        rec = _make_record(name, dt, fields)
        _emit_perf(rec)
        logger.info(f"{name} {dt:.3f}s {fields if fields else ''}")


@contextlib.contextmanager
def _collect_block(records, name, **fields):
    """Like time_block but appends the record to `records` instead of emitting.

    Used inside joblib workers so the parent process can serialize log writes
    after results return — avoids cross-process file-handle contention.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        records.append(_make_record(name, dt, fields))


def _load_csv_cached(csv_path, kind, case_id, cohort, perf_records):
    """Read a per-case CSV with on-disk .npy caching.

    Cache invalidates by mtime — if the underlying CSV is newer than the cache,
    we reparse. Atomic via os.replace to keep partial writes off disk.
    Opt-out with `CONS_NO_CACHE=1`.
    """
    if _CACHE_DISABLED:
        with _collect_block(perf_records, "csv_parse", case=case_id, kind=kind):
            return pd.read_csv(csv_path, header=None).values

    cache_dir = os.path.join(_CACHE_DIR, cohort)
    npy_path = os.path.join(cache_dir, f"{case_id}_{kind}.npy")
    try:
        if os.path.exists(npy_path) and os.path.getmtime(npy_path) >= os.path.getmtime(
            csv_path
        ):
            with _collect_block(perf_records, "csv_cache_hit", case=case_id, kind=kind):
                return numpy.load(npy_path)
    except OSError:
        pass

    with _collect_block(perf_records, "csv_parse", case=case_id, kind=kind):
        arr = pd.read_csv(csv_path, header=None).values

    try:
        os.makedirs(cache_dir, exist_ok=True)
        # Write atomically so a crash mid-save can't leave a half-written cache
        # behind. The tmp path ends in `.npy` so numpy.save doesn't append a
        # second `.npy` suffix and confuse the os.replace target.
        tmp_path = npy_path[:-4] + ".tmp.npy"
        numpy.save(tmp_path, arr)
        os.replace(tmp_path, npy_path)
    except OSError as e:
        logger.warning(f"cache write failed for {npy_path}: {e}")

    return arr


with open(os.path.join(_REPO_ROOT, "dataset/OR/rx_sorted_case_ids.yml")) as _rx_fh:
    _RX_BY_CASE = {
        cid: cmpd for cmpd, cids in yaml.safe_load(_rx_fh).items() for cid in cids
    }


def identify_compound(case_id):
    try:
        return _RX_BY_CASE[case_id]
    except KeyError as exc:
        raise ValueError(f"invalid case id: {case_id}") from exc


def _process_one_case(filepath):
    """Process a single `*_Sdb.csv` and its sibling files into 0–2 result rows.

    Returns (results, perf_records). All timings are collected into perf_records
    and emitted by the parent process to keep file writes serialized.
    """
    perf_records = []
    results = []
    fname = os.path.basename(filepath)
    case_id = fname.split("_")[0]
    cohort = os.path.basename(os.path.dirname(filepath))
    base_path = os.path.dirname(filepath)

    case_t0 = time.perf_counter()
    try:
        sdb = _load_csv_cached(filepath, "Sdb", case_id, cohort, perf_records)
        f = _load_csv_cached(
            os.path.join(base_path, f"{case_id}_f.csv"),
            "f",
            case_id,
            cohort,
            perf_records,
        ).flatten()

        l_path = os.path.join(base_path, f"{case_id}_l.csv")
        if not os.path.exists(l_path):
            return results, perf_records
        labels = _load_csv_cached(l_path, "l", case_id, cohort, perf_records).flatten()

        P = 10 ** (sdb / 10)
        if P.shape[0] == len(f):
            alpha = P[(f >= 8) & (f <= 12), :].mean(axis=0)
            delta = P[(f >= 1) & (f <= 4), :].mean(axis=0)
            theta = P[(f >= 4) & (f <= 8), :].mean(axis=0)
            beta = P[(f >= 13) & (f <= 30), :].mean(axis=0)
            gamma = P[(f >= 30) & (f <= 45), :].mean(axis=0)
        else:
            alpha = P[:, (f >= 8) & (f <= 12)].mean(axis=1)
            delta = P[:, (f >= 1) & (f <= 4)].mean(axis=1)
            theta = P[:, (f >= 4) & (f <= 8)].mean(axis=1)
            beta = P[:, (f >= 13) & (f <= 30)].mean(axis=1)
            gamma = P[:, (f >= 30) & (f <= 45)].mean(axis=1)

        min_len = min(len(alpha), len(delta), len(labels))

        quality_mask = numpy.ones(min_len, dtype=bool)
        if cohort == "OR":
            q_path = os.path.join(base_path, f"{case_id}_EEGquality.csv")
            if os.path.exists(q_path):
                q = _load_csv_cached(
                    q_path, "EEGquality", case_id, cohort, perf_records
                ).flatten()
                min_len = min(min_len, len(q))
                quality_mask = q[:min_len].astype(bool)

        alpha = alpha[:min_len]
        delta = delta[:min_len]
        theta = theta[:min_len]
        beta = beta[:min_len]
        gamma = gamma[:min_len]
        labels = labels[:min_len]
        quality_mask = quality_mask[:min_len]

        compound = identify_compound(case_id) if cohort == "OR" else "pure_propofol"

        for i in range(0, min_len - WINDOW_BINS + 1, WINDOW_STRIDE):
            sl = slice(i, i + WINDOW_BINS)
            if quality_mask[sl].mean() < QUALITY_MIN_FRAC:
                continue
            lab_win = labels[sl]
            if lab_win.min() != lab_win.max():
                continue
            alpha_win = alpha[sl]
            if numpy.std(alpha_win) == 0:
                continue
            delta_win = delta[sl]

            with _collect_block(
                perf_records, "epoch_metrics", case=case_id, epoch_id=i
            ):
                LZ_classic = eeg.lz_classic_binary(alpha_win)
                POLZ_4 = eeg.polz_complexity(alpha_win, 4)
                POLZ_8 = eeg.polz_complexity(alpha_win, 8)
                POLZ_16 = eeg.polz_complexity(alpha_win, 16)
                POLZ_32 = eeg.polz_complexity(alpha_win, 32) 
                K = eeg.median_K(alpha_win)
                alpha_mean = numpy.mean(alpha_win)
                delta_mean = numpy.mean(delta_win)
                log_ratio = float(
                    numpy.log(delta_mean + 1e-12) - numpy.log(alpha_mean + 1e-12)
                )
                log_alpha_var = float(numpy.log(numpy.var(alpha_win) + 1e-12))
                if P.shape[0] == len(f):
                    spec = P[:, sl].mean(axis=1)
                else:
                    spec = P[sl, :].mean(axis=0)
                spec = spec / (spec.sum() + 1e-12)
                spectral_entropy = float(-(spec * numpy.log(spec + 1e-12)).sum())

                results.append(
                    {
                        "case": case_id,
                        "state": "Conscious" if lab_win[0] == 1 else "Unconscious",
                        "epoch_number": i,
                        "K": K,
                        "LZ_Classic": LZ_classic,
                        "POLZ_4": POLZ_4,
                        "POLZ_8": POLZ_8,
                        "POLZ_16": POLZ_16,
                        "POLZ_32": POLZ_32,
                        "log_ratio": log_ratio,
                        "log_alpha_var": log_alpha_var,
                        "log_theta": float(numpy.log(theta[sl].mean() + 1e-12)),
                        "log_beta": float(numpy.log(beta[sl].mean() + 1e-12)),
                        "log_gamma": float(numpy.log(gamma[sl].mean() + 1e-12)),
                        "spectral_entropy": spectral_entropy,
                        "n_samples": min_len,
                        "compound": compound,
                        "cohort": cohort,
                    }
                )

    except Exception as e:
        # Surface the failure in perf log so it isn't silent
        print(f"ERROR{e}")
        perf_records.append(
            _make_record(
                "case_error",
                time.perf_counter() - case_t0,
                {"case": case_id, "error": repr(e)},
            )
        )

    perf_records.append(
        _make_record(
            "case",
            time.perf_counter() - case_t0,
            {"case": case_id, "cohort": cohort, "rows": len(results)},
        )
    )
    return results, perf_records


def load_data():
    with time_block("load_data_total"):
        files = glob.glob("dataset/**/*_Sdb.csv", recursive=True)
        print(
            f"Found {len(files)} files. Processing by State (Conscious vs Unconscious)..."
        )

        # Trigger numba JIT compile in the parent before forking workers; each
        # loky worker still recompiles on first call (logged separately).
        with time_block("numba_warmup_parent"):
            eeg.lz_fast(numpy.zeros(8, dtype=numpy.int8))

        with time_block("file_dispatch", n_files=len(files)):
            out = Parallel(n_jobs=-1, backend="loky")(
                delayed(_process_one_case)(fp) for fp in files
            )

        results = []
        for case_results, case_perf in out:
            results.extend(case_results)
            for rec in case_perf:
                _emit_perf(rec)

        df = pd.DataFrame(results)
        df["label"] = df["state"].map({"Conscious": 1, "Unconscious": 0})

        X = df[FEATURE_COLS].values
        y = df["label"].values
        groups = df["case"].values

    return X, y, df, groups


def plot_scatter(ax, df, feat_x="K", feat_y="LZ"):
    conscious = df[df["label"] == 1]
    unconscious = df[df["label"] == 0]

    is_outlier = df["n_samples"] <= OUTLIER_THRESHOLD

    c_ok = conscious[~is_outlier[conscious.index]]
    c_short = conscious[is_outlier[conscious.index]]
    u_ok = unconscious[~is_outlier[unconscious.index]]
    u_short = unconscious[is_outlier[unconscious.index]]

    ax.scatter(
        c_ok[feat_x], c_ok[feat_y], color="red", label="Conscious", edgecolor="k", s=50
    )
    ax.scatter(
        u_ok[feat_x], u_ok[feat_y], color="blue", label="Unconscious", marker="x", s=50
    )

    if len(c_short) > 0:
        ax.plot(
            c_short[feat_x].values,
            c_short[feat_y].values,
            "o",
            color="red",
            markerfacecoloralt="yellow",
            fillstyle="left",
            markersize=7,
            markeredgecolor="k",
            label=f"Conscious (n\u2264{OUTLIER_THRESHOLD})",
            linestyle="None",
        )
    if len(u_short) > 0:
        ax.plot(
            u_short[feat_x].values,
            u_short[feat_y].values,
            "s",
            color="blue",
            markerfacecoloralt="yellow",
            fillstyle="left",
            markersize=7,
            markeredgecolor="k",
            label=f"Unconscious (n\u2264{OUTLIER_THRESHOLD})",
            linestyle="None",
        )


def tuning_svm(X, y, groups, gpu=False):
    inner_cv = GroupKFold(n_splits=5)

    if gpu:
        from cuml.preprocessing import RobustScaler as cuRobustScaler
        from cuml.svm import SVC as cuSVC
        import cuml

        cuml.set_global_output_type("numpy")
        param_dist = {
            "svc__C": loguniform(0.1, 1000),
            "svc__gamma": loguniform(1e-4, 1),
        }
        pipeline_svm = make_pipeline(
            cuRobustScaler(),
            cuSVC(kernel="rbf", class_weight="balanced", probability=False),
        )
        n_jobs = 1
    else:
        param_dist = {
            "svc__C": loguniform(0.1, 1000),
            "svc__gamma": loguniform(1e-4, 1),
        }
        pipeline_svm = make_pipeline(
            PowerTransformer(method="yeo-johnson", standardize=True),
            SVC(kernel="rbf", class_weight="balanced", probability=False),
        )
        n_jobs = -1

    with time_block("tuning_svm", n_iter=60, gpu=gpu):
        search = RandomizedSearchCV(
            pipeline_svm,
            param_dist,
            n_iter=60,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=n_jobs,
            random_state=42,
        ).fit(X, y, groups=groups)

    print(f"[SVM] Best parameters:: {search.best_params_}")

    return search.best_estimator_


C_COLOR = "#c44e52"  # Conscious (red)
U_COLOR = "#4c72b0"  # Unconscious (blue)


def _univariate_separation(df):
    """Per-feature univariate AUC (max(AUC, 1-AUC)), highest first.

    Used to choose the two most discriminative features for the joint plot and
    to order the per-feature panel.
    """
    y = df["label"].values
    scores = {}
    for f in FEATURE_COLS:
        v = df[f].values
        if numpy.std(v) == 0:
            scores[f] = 0.5
            continue
        a = roc_auc_score(y, v)
        scores[f] = max(a, 1 - a)
    return sorted(FEATURE_COLS, key=lambda f: scores[f], reverse=True), scores


def _clipped_range(v, lo=1, hi=99, pad=0.05):
    a, b = numpy.percentile(v, [lo, hi])
    m = (b - a) * pad
    return a - m, b + m


def _kde_contours(ax, x, y, xs, ys, color):
    """Density contours of (x, y): a transparent->color gradient fill (so
    low-density regions stay clear instead of washing out the whole panel),
    plus line contours.
    """
    try:
        k = gaussian_kde(numpy.vstack([x, y]))
    except numpy.linalg.LinAlgError:
        return
    XX, YY = numpy.meshgrid(xs, ys)
    ZZ = k(numpy.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    rgb = to_rgb(color)
    cmap = LinearSegmentedColormap.from_list("", [(*rgb, 0.0), (*rgb, 0.55)])
    # Start fills above ~7% of peak density so the background stays white.
    levels = numpy.linspace(ZZ.max() * 0.07, ZZ.max(), 7)
    ax.contourf(XX, YY, ZZ, levels=levels, cmap=cmap, extend="max")
    ax.contour(XX, YY, ZZ, levels=levels, colors=[color], linewidths=0.8, alpha=0.85)


def plot_feature_overview(df, subtitle="", fx="K", fy="LZ"):
    """Cleaner replacement for the dense 3D scatter.

    Per-class 2D density contours on two features (default: the complexity
    primitives K and LZ) with marginal densities — overplotting-free and
    unaffected by class imbalance since each class density is self-normalized —
    over the decision boundary of an RBF-SVM fit on those two features.
    """
    _, uni = _univariate_separation(df)
    is_c = df["label"].values == 1

    fig = plt.figure(figsize=(8.5, 8))
    jb = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.04, hspace=0.04
    )
    ax = fig.add_subplot(jb[1, 0])
    ax_top = fig.add_subplot(jb[0, 0], sharex=ax)
    ax_right = fig.add_subplot(jb[1, 1], sharey=ax)

    xv, yv = df[fx].values, df[fy].values
    x0, x1 = _clipped_range(xv)
    y0, y1 = _clipped_range(yv)
    xs = numpy.linspace(x0, x1, 200)
    ys = numpy.linspace(y0, y1, 200)

    # Decision boundary of an RBF-SVM trained on *only* these two features — a 2D
    # view of the classifier (the production model uses all 8 features, so its
    # true boundary lives in 8D and can't be drawn here). Shaded regions show the
    # predicted class; the dashed line is the decision_function == 0 contour.
    XX, YY = numpy.meshgrid(xs, ys)
    grid = numpy.column_stack([XX.ravel(), YY.ravel()])
    disp_clf = make_pipeline(
        PowerTransformer(method="yeo-johnson", standardize=True),
        SVC(kernel="rbf", class_weight="balanced", gamma="scale", C=10.0),
    ).fit(df[[fx, fy]].values, df["label"].values)
    ZZc = disp_clf.decision_function(grid).reshape(XX.shape)
    ax.contourf(
        XX, YY, ZZc, levels=[-1e18, 0, 1e18], colors=[U_COLOR, C_COLOR], alpha=0.08
    )
    ax.contour(XX, YY, ZZc, levels=[0], colors="k", linewidths=1.5, linestyles="--")
    ax.plot([], [], "k--", lw=1.5, label="SVM boundary (2-feature)")

    for mask, color, label in [
        (is_c, C_COLOR, "Conscious"),
        (~is_c, U_COLOR, "Unconscious"),
    ]:
        _kde_contours(ax, xv[mask], yv[mask], xs, ys, color)
        kx = gaussian_kde(xv[mask])
        ax_top.fill_between(xs, kx(xs), color=color, alpha=0.4)
        ax_top.plot(xs, kx(xs), color=color, lw=1.2)
        ky = gaussian_kde(yv[mask])
        ax_right.fill_betweenx(ys, ky(ys), color=color, alpha=0.4)
        ax_right.plot(ky(ys), ys, color=color, lw=1.2)
        ax.plot([], [], color=color, lw=6, alpha=0.5, label=label)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(f"{fx}  (univariate AUC {uni[fx]:.2f})")
    ax.set_ylabel(f"{fy}  (univariate AUC {uni[fy]:.2f})")
    ax.legend(loc="upper right", frameon=True)
    ax_top.axis("off")
    ax_right.axis("off")

    title = "Conscious vs Unconscious — feature density"
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.92)


def _bca_ci(boot_thetas, jack_thetas, theta_hat, alpha=0.05):
    """Bias-corrected and accelerated (BCa) bootstrap interval."""
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


def _auc_with_ci(y, score, groups, n_boot=2000, seed=42):
    """Pooled ROC-AUC plus a case-level cluster-bootstrap BCa 95% CI."""
    y = numpy.asarray(y)
    score = numpy.asarray(score)
    groups = numpy.asarray(groups)

    def _auc(yt, ys):
        if len(numpy.unique(yt)) < 2:
            return float("nan")
        return roc_auc_score(yt, ys)

    theta_hat = _auc(y, score)

    unique = numpy.unique(groups)
    case_rows = {c: numpy.flatnonzero(groups == c) for c in unique}
    rng = numpy.random.default_rng(seed)

    boots = numpy.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(unique, size=len(unique), replace=True)
        rows = numpy.concatenate([case_rows[c] for c in sample])
        boots[b] = _auc(y[rows], score[rows])

    jacks = numpy.empty(len(unique))
    for i, c in enumerate(unique):
        rows = numpy.concatenate([case_rows[cc] for cc in unique if cc != c])
        jacks[i] = _auc(y[rows], score[rows])

    lo, hi = _bca_ci(boots, jacks, theta_hat)
    return theta_hat, lo, hi


def plot_roc_overview(y, oof, slices, agg):
    """Compact ROC panel on the model's pooled out-of-fold predictions."""
    y = numpy.asarray(y)
    oof = numpy.asarray(oof)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    summary = []

    case_auc, case_lo, case_hi = _auc_with_ci(
        agg["label"].values, agg["score"].values, agg["case"].values
    )
    fpr, tpr, _ = roc_curve(agg["label"].values, agg["score"].values)
    ax.plot(
        fpr,
        tpr,
        color="black",
        lw=2.4,
        linestyle="--",
        zorder=5,
        label=f"Case-level (aggregated) - {case_auc:.2f} [{case_lo:.2f}, {case_hi:.2f}]",
    )
    summary.append(("case-level", case_auc, case_lo, case_hi))

    for label, mask, color, groups, style in slices:
        m = numpy.asarray(mask, dtype=bool)
        if m.sum() == 0 or len(numpy.unique(y[m])) < 2:
            continue
        auc, lo, hi = _auc_with_ci(y[m], oof[m], numpy.asarray(groups)[m])
        fpr, tpr, _ = roc_curve(y[m], oof[m])
        ax.plot(
            fpr,
            tpr,
            color=color,
            label=f"{label} - {auc:.2f} [{lo:.2f}, {hi:.2f}]",
            **style,
        )
        summary.append((label, auc, lo, hi))

    ax.plot(
        [0, 1], [0, 1], color="grey", lw=1.0, linestyle=":", zorder=0, label="chance"
    )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Conscious vs Unconscious - ROC (pooled out-of-fold)")
    ax.legend(loc="lower right", frameon=True, fontsize=7.5, handlelength=1.6)
    fig.tight_layout()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use cuML GPU SVM (requires cuml installed; drops Nystroem, uses exact RBF)",
    )
    args = parser.parse_args()

    with time_block("total_run", gpu=args.gpu):
        X_full, y, df, groups = load_data()
        cv = GroupKFold(n_splits=5)
        
        # Máscaras de filtros para extrair as métricas detalhadas depois
        propofol_mask = (df["compound"] == "pure_propofol").values
        sevoflurane_mask = df["compound"].isin(["mixed", "pure_sevo"]).values

        print(f"Total samples (lines): {len(df)}")
        print(f"Unique patients: {len(numpy.unique(groups))}")
        print(f"Conscious: {(y == 1).sum()}, Unconscious: {(y == 0).sum()}")

        BASE_FEATURES = [
            "K", "log_ratio", "log_alpha_var", "log_theta", 
            "log_beta", "log_gamma", "spectral_entropy"
        ]

        methodologies = {
            "SVM + Clássico (2)": ["LZ_Classic"] + BASE_FEATURES,
            "SVM + POLZ (4)": ["POLZ_4"] + BASE_FEATURES,
            "SVM + POLZ (8)": ["POLZ_8"] + BASE_FEATURES,
            "SVM + POLZ (16)": ["POLZ_16"] + BASE_FEATURES,
            "SVM + POLZ (32)": ["POLZ_32"] + BASE_FEATURES, 
        }

        models = {}
        for name, features in methodologies.items():
            print(f"\n--- Otimizando {name} ---")
            X_sub = df[features].values
            
            if args.gpu:
                X_sub = numpy.ascontiguousarray(X_sub, dtype=numpy.float32)
                y = numpy.ascontiguousarray(y, dtype=numpy.int32)
                
            svm_tuned = tuning_svm(X_sub, y, groups, gpu=args.gpu)
            models[name] = (svm_tuned, features)

        auc_scores = {name: [] for name in models}
        acc_scores = {name: [] for name in models} # Para guardar a acurácia de cada fold
        oof_scores = {name: numpy.full(len(y), numpy.nan) for name in models}

        with time_block("auc_loop_total", n_folds=5, n_models=len(models)):
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df, y, groups)):
                y_train, y_test = y[train_idx], y[test_idx]

                for name, (model, features) in models.items():
                    X_sub = df[features].values
                    X_train, X_test = X_sub[train_idx], X_sub[test_idx]

                    with time_block("cv_fold", fold=fold_idx, model=name):
                        model.fit(X_train, y_train)
                        
                        # 1. Pega os Scores (Probabilidade ou Distância da Fronteira) para AUC
                        if hasattr(model, "predict_proba"):
                            y_score = model.predict_proba(X_test)[:, 1]
                        elif hasattr(model, "decision_function"):
                            y_score = model.decision_function(X_test)
                        
                        # 2. Pega a predição da classe (0 ou 1) para Acurácia
                        y_pred = model.predict(X_test)
                            
                        auc = roc_auc_score(y_test, y_score)
                        acc = (y_test == y_pred).mean() # Cálculo direto de acurácia
                        
                        auc_scores[name].append(auc)
                        acc_scores[name].append(acc)
                        oof_scores[name][test_idx] = y_score

        # --- IMPRESSÃO DOS RESULTADOS DETALHADOS ---
        print("\n" + "="*70)
        print("MÉTRICAS DETALHADAS POR METODOLOGIA".center(70))
        print("="*70)
        
        aucs_for_plot = {}
        for name in models.keys():
            print(f"\n[ Resultados: {name} ]")
            
            # --- CV Accuracy (Média e Desvio Padrão) ---
            acc_mean = numpy.mean(acc_scores[name]) * 100
            acc_std = numpy.std(acc_scores[name]) * 100
            print(f"{name}: 3D CV Accuracy = {acc_mean:.2f}% (+/- {acc_std:.2f}%)")
            print(f"{name}: window-level median CV AUC = {numpy.median(auc_scores[name]):.4f}")
            
            # --- Case-Level AUC ---
            agg = (
                df.assign(score=oof_scores[name])
                .groupby(["case", "state"], as_index=False)
                .agg(score=("score", "mean"), label=("label", "first"))
            )
            case_auc, case_lo, case_hi = _auc_with_ci(
                agg["label"].values, agg["score"].values, agg["case"].values
            )
            print(f"{name}: pooled-OOF case-level AUC = {case_auc:.4f} [{case_lo:.4f}, {case_hi:.4f}]")
            
            # --- Overall (Window) AUC ---
            over_auc, over_lo, over_hi = _auc_with_ci(y, oof_scores[name], groups)
            print(f"{name}: pooled-OOF overall (window) AUC = {over_auc:.4f} [{over_lo:.4f}, {over_hi:.4f}]")
            
            # --- Propofol (Window) AUC ---
            prop_auc, prop_lo, prop_hi = _auc_with_ci(
                y[propofol_mask], oof_scores[name][propofol_mask], groups[propofol_mask]
            )
            print(f"{name}: pooled-OOF propofol (window) AUC = {prop_auc:.4f} [{prop_lo:.4f}, {prop_hi:.4f}]")
            
            # --- Sevoflurane (Window) AUC ---
            sevo_auc, sevo_lo, sevo_hi = _auc_with_ci(
                y[sevoflurane_mask], oof_scores[name][sevoflurane_mask], groups[sevoflurane_mask]
            )
            print(f"{name}: pooled-OOF sevoflurane (window) AUC = {sevo_auc:.4f} [{sevo_lo:.4f}, {sevo_hi:.4f}]")
            
            # Guarda para o plot comparativo
            aucs_for_plot[name] = {"auc": case_auc, "lo": case_lo, "hi": case_hi}

        # --- GERAÇÃO DOS GRÁFICOS ---
        with time_block("plot_svm_comparisons"):
            img_dir = os.path.join(_REPO_ROOT, "images")
            os.makedirs(img_dir, exist_ok=True)
            colors = ['#7f7f7f', '#4c72b0', '#55a868', '#c44e52', '#8172b3']
            
            # PLOT 1: Curva ROC
            fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
            for (name, color) in zip(models.keys(), colors):
                agg = (
                    df.assign(score=oof_scores[name])
                    .groupby(["case", "state"], as_index=False)
                    .agg(score=("score", "mean"), label=("label", "first"))
                )
                fpr, tpr, _ = roc_curve(agg["label"], agg["score"])
                auc_val = aucs_for_plot[name]["auc"]
                ax_roc.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC: {auc_val:.3f})")
            
            ax_roc.plot([0, 1], [0, 1], color="black", lw=1.5, linestyle="--", label="Chance")
            ax_roc.set_xlim(-0.01, 1.01)
            ax_roc.set_ylim(-0.01, 1.01)
            ax_roc.set_aspect("equal")
            ax_roc.set_xlabel("Taxa de Falsos Positivos")
            ax_roc.set_ylabel("Taxa de Verdadeiros Positivos")
            ax_roc.set_title("Curvas ROC Comparativas (Nível do Paciente)")
            ax_roc.legend(loc="lower right", frameon=True)
            
            roc_fig_path = os.path.join(img_dir, "svm_alphabet_comparison_roc.png")
            fig_roc.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
            
            # PLOT 2: Gráfico de Intervalos de Confiança (Forest Plot-style)
            fig_ci, ax_ci = plt.subplots(figsize=(9, 6))
            model_names = list(aucs_for_plot.keys())
            
            auc_vals = [aucs_for_plot[m]["auc"] for m in model_names]
            err_lo = [aucs_for_plot[m]["auc"] - aucs_for_plot[m]["lo"] for m in model_names]
            err_hi = [aucs_for_plot[m]["hi"] - aucs_for_plot[m]["auc"] for m in model_names]
            
            for i, name in enumerate(model_names):
                ax_ci.errorbar(
                    name, auc_vals[i], 
                    yerr=[[err_lo[i]], [err_hi[i]]],
                    fmt='o', color=colors[i], ecolor='black', 
                    capsize=8, markersize=10, markeredgewidth=2.5, lw=2.5
                )

            ax_ci.set_ylabel("ROC-AUC do Paciente (95% CI Bootstrap)")
            ax_ci.set_title("Comparação de Desempenho e IC por Tamanho do Alfabeto", fontsize=13)
            ax_ci.grid(axis='y', linestyle='--', alpha=0.6)
            
            plt.setp(ax_ci.get_xticklabels(), rotation=15, ha="right")
            
            ci_fig_path = os.path.join(img_dir, "svm_alphabet_confidence_intervals.png")
            fig_ci.savefig(ci_fig_path, dpi=300, bbox_inches="tight")
            
            print(f"\nSalvo gráfico comparativo ROC em {roc_fig_path}")
            print(f"Salvo gráfico de Intervalos de Confiança em {ci_fig_path}")

    plt.show()

if __name__ == "__main__":
    main()


