import argparse
import os

import matplotlib.pyplot as plt
import numpy
import pandas as pd
from scipy.stats import loguniform
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

import analysis as A

_OUT_DIR = os.path.join(A._REPO_ROOT, "stats_results")
os.makedirs(_OUT_DIR, exist_ok=True)
_IMG_DIR = os.path.join(A._REPO_ROOT, "images")
os.makedirs(_IMG_DIR, exist_ok=True)
_CURVES_PATH = os.path.join(_OUT_DIR, "abel_comparison_curves.npz")

SLICE_ORDER = ["overall", "propofol", "sevo", "volunteers"]
SLICE_LABEL = {
    "overall": "Overall",
    "propofol": "Propofol",
    "sevo": "Sevoflurane",
    "volunteers": "Volunteers",
}

PALETTE = ["#c44e52", "#4c72b0", "#55a868", "#8172b3"]
LINE_STYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D"]

_OURS_DIST = {"svc__C": loguniform(0.1, 1000), "svc__gamma": loguniform(1e-4, 1)}


def slice_masks(frame):
    return {
        "overall": numpy.ones(len(frame), dtype=bool),
        "propofol": (frame["compound"] == "pure_propofol").values,
        "sevo": frame["compound"].isin(["mixed", "pure_sevo"]).values,
        "volunteers": (frame["cohort"] != "OR").values,
    }


def load_perbin(df):
    meta = df.groupby("case")[["cohort", "compound"]].first()
    X_parts, y_parts, frames = [], [], []
    for case, row in meta.iterrows():
        cohort, compound = row["cohort"], row["compound"]
        base = os.path.join("dataset", cohort)
        sdb = A._load_csv_cached(
            os.path.join(base, f"{case}_Sdb.csv"), "Sdb", case, cohort, []
        )
        f = A._load_csv_cached(
            os.path.join(base, f"{case}_f.csv"), "f", case, cohort, []
        ).flatten()
        labels = A._load_csv_cached(
            os.path.join(base, f"{case}_l.csv"), "l", case, cohort, []
        ).flatten()
        t = A._load_csv_cached(
            os.path.join(base, f"{case}_t.csv"), "t", case, cohort, []
        ).flatten()

        spec = sdb.T if sdb.shape[0] == len(f) else sdb
        n = min(spec.shape[0], len(labels), len(t))
        spec, labels, t = spec[:n], labels[:n], t[:n]

        egq = numpy.ones(n, dtype=bool)
        if cohort == "OR":
            qp = os.path.join(base, f"{case}_EEGquality.csv")
            if os.path.exists(qp):
                q = A._load_csv_cached(qp, "EEGquality", case, cohort, []).flatten()
                m = min(n, len(q))
                spec, labels, t, egq = spec[:m], labels[:m], t[:m], q[:m].astype(bool)

        keep = egq & numpy.isin(labels, [0, 1])
        spec, labels, t = spec[keep], labels[keep], t[keep]
        if len(labels) == 0:
            continue

        X_parts.append(spec)
        y_parts.append(labels.astype(int))
        frames.append(
            pd.DataFrame(
                {
                    "case": case,
                    "cohort": cohort,
                    "compound": compound,
                    "t": t,
                }
            )
        )

    X = numpy.vstack(X_parts)
    y = numpy.concatenate(y_parts)
    bins = pd.concat(frames, ignore_index=True)
    return X, y, bins


def _order_and_lengths(idx, bins):
    sub = bins.loc[idx, ["case", "t"]].sort_values(["case", "t"])
    order = sub.index.values
    lengths = []
    for _, g in sub.groupby("case", sort=False):
        tv = g["t"].values
        dt = numpy.median(numpy.diff(tv)) if len(tv) > 1 else 2.0
        run = 1
        for a, b in zip(tv[:-1], tv[1:]):
            if b - a <= 1.5 * dt:
                run += 1
            else:
                lengths.append(run)
                run = 1
        lengths.append(run)
    return order, numpy.asarray(lengths, dtype=int)


def _make_ours():
    return make_pipeline(
        PowerTransformer(method="yeo-johnson", standardize=True),
        SVC(kernel="rbf", class_weight="balanced", probability=False),
    )


def _make_abel_sdb():
    return LogisticRegression(solver="liblinear", max_iter=1000)


def _make_abel_pca():
    return make_pipeline(PCA(n_components=3), LogisticRegression(max_iter=1000))


def _tuned_oof(make_est, param_dist, X, y, groups, tr, te, n_iter, seed, proba):
    search = RandomizedSearchCV(
        make_est(),
        param_dist,
        n_iter=n_iter,
        cv=GroupKFold(5),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=seed,
    ).fit(X[tr], y[tr], groups=groups[tr])
    est = search.best_estimator_
    score = est.predict_proba(X[te])[:, 1] if proba else est.decision_function(X[te])
    return score, search.best_params_


def _lda_hmm2_perbin(X, y, bins, tr, te):
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X[tr], y[tr])
    ld = lda.transform(X)[:, 0]

    try:
        from hmmlearn.hmm import GaussianHMM

        order_tr, len_tr = _order_and_lengths(tr, bins)
        order_te, len_te = _order_and_lengths(te, bins)
        hmm = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            algorithm="viterbi",
            n_iter=10,
            random_state=42,
        )
        hmm.fit(ld[order_tr].reshape(-1, 1), len_tr)
        post_tr = hmm.predict_proba(ld[order_tr].reshape(-1, 1), len_tr)
        post_te = hmm.predict_proba(ld[order_te].reshape(-1, 1), len_te)

        lr = LogisticRegression(max_iter=5000)
        lr.fit(post_tr, y[order_tr])
        scores = numpy.full(len(y), numpy.nan)
        scores[order_te] = lr.predict_proba(post_te)[:, 1]
        return scores, True
    except Exception as e:
        print(f"  hmm fit failed ({e!r}), falling back to plain LDA-LR")

    lr = LogisticRegression(max_iter=5000)
    lr.fit(ld[tr].reshape(-1, 1), y[tr])
    scores = numpy.full(len(y), numpy.nan)
    scores[te] = lr.predict_proba(ld[te].reshape(-1, 1))[:, 1]
    return scores, False


def run_comparison(args):
    X8, y, df, groups = A.load_data()
    df = df.reset_index(drop=True)
    X8 = df[A.FEATURE_COLS].values
    y = df["label"].values.astype(int)
    groups = df["case"].values
    masks = slice_masks(df)

    Xb, yb, bins = load_perbin(df)
    gb = bins["case"].values
    masks_b = slice_masks(bins)

    print(
        f"\nours: {len(df)} epochs over {len(numpy.unique(groups))} cases  |  "
        f"abel: {len(bins)} bins over {len(numpy.unique(gb))} cases"
    )

    cv = GroupKFold(n_splits=5)

    oof_ours = numpy.full(len(y), numpy.nan)
    for fold, (tr, te) in enumerate(cv.split(X8, y, groups)):
        s, params = _tuned_oof(
            _make_ours,
            _OURS_DIST,
            X8,
            y,
            groups,
            tr,
            te,
            args.n_iter,
            args.seed,
            proba=False,
        )
        oof_ours[te] = s
        print(f"  fold {fold + 1}/5  {params}")

    abel_names = ["Abel Sdb", "Abel PCA"] + ([] if args.no_hmm else ["Abel LDA+HMM2"])
    oof_b = {n: numpy.full(len(yb), numpy.nan) for n in abel_names}
    hmm_used_any = False
    for tr, te in cv.split(Xb, yb, gb):
        oof_b["Abel Sdb"][te] = (
            _make_abel_sdb().fit(Xb[tr], yb[tr]).predict_proba(Xb[te])[:, 1]
        )
        oof_b["Abel PCA"][te] = (
            _make_abel_pca().fit(Xb[tr], yb[tr]).predict_proba(Xb[te])[:, 1]
        )
        if not args.no_hmm:
            s, used = _lda_hmm2_perbin(Xb, yb, bins, tr, te)
            mask = ~numpy.isnan(s)
            oof_b["Abel LDA+HMM2"][mask] = s[mask]
            hmm_used_any = hmm_used_any or used

    if not args.no_hmm and not hmm_used_any:
        oof_b["Abel LDA-LR (HMM failed)"] = oof_b.pop("Abel LDA+HMM2")
        abel_names[abel_names.index("Abel LDA+HMM2")] = "Abel LDA-LR (HMM failed)"

    curves = {"Ours (8-feat SVM)": (y, oof_ours, groups, masks)}
    for name in abel_names:
        curves[name] = (yb, oof_b[name], gb, masks_b)
    model_names = ["Ours (8-feat SVM)"] + abel_names

    rows = []
    for name in model_names:
        yy, ss, gg, mm = curves[name]
        for sl in SLICE_ORDER:
            mask = mm[sl]
            auc, lo, hi = A._auc_with_ci(
                yy[mask], ss[mask], gg[mask], n_boot=args.n_boot
            )
            rows.append(
                {"model": name, "slice": sl, "auc": auc, "ci_low": lo, "ci_high": hi}
            )
    res = pd.DataFrame(rows)
    csv_path = os.path.join(_OUT_DIR, "abel_comparison_auc.csv")
    res.to_csv(csv_path, index=False)

    print("\npooled OOF auc [bca 95% ci]:")
    print(f"{'model':22s} " + " ".join(f"{SLICE_LABEL[s]:>22s}" for s in SLICE_ORDER))
    for name in model_names:
        cells = []
        for sl in SLICE_ORDER:
            r = res[(res.model == name) & (res.slice == sl)].iloc[0]
            cells.append(f"{r.auc:.3f} [{r.ci_low:.2f},{r.ci_high:.2f}]")
        print(f"{name:22s} " + " ".join(f"{c:>22s}" for c in cells))

    _save_curves(curves, model_names)
    _plot(curves, res, model_names)
    print(f"wrote {csv_path}")
    return res


def _save_curves(curves, model_names):
    arrs = {
        "__models__": numpy.asarray(model_names),
        "__slices__": numpy.asarray(SLICE_ORDER),
    }
    for i, name in enumerate(model_names):
        y, score, groups, masks = curves[name]
        arrs[f"m{i}_y"] = numpy.asarray(y)
        arrs[f"m{i}_score"] = numpy.asarray(score, dtype=float)
        arrs[f"m{i}_groups"] = numpy.asarray(groups, dtype=str)
        for sl in SLICE_ORDER:
            arrs[f"m{i}_mask_{sl}"] = numpy.asarray(masks[sl], dtype=bool)
    numpy.savez_compressed(_CURVES_PATH, **arrs)


def _load_curves():
    d = numpy.load(_CURVES_PATH, allow_pickle=False)
    model_names = [str(x) for x in d["__models__"]]
    slices = [str(x) for x in d["__slices__"]]
    curves = {}
    for i, name in enumerate(model_names):
        masks = {sl: d[f"m{i}_mask_{sl}"] for sl in slices}
        curves[name] = (d[f"m{i}_y"], d[f"m{i}_score"], d[f"m{i}_groups"], masks)
    return curves, model_names


def _plot_auc_by_cohort(res, model_names, out_name):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    n = len(model_names)
    width = 0.17
    conditions = SLICE_ORDER
    xs = numpy.arange(len(conditions), dtype=float)

    for j in range(len(conditions)):
        if j % 2 == 0:
            ax.axvspan(j - 0.5, j + 0.5, color="0.92", zorder=0)

    def cell(name, sl):
        r = res[(res.model == name) & (res.slice == sl)].iloc[0]
        return r.auc, r.auc - r.ci_low, r.ci_high - r.auc

    for i, name in enumerate(model_names):
        offs = (i - (n - 1) / 2) * width
        aucs, los, his = [], [], []
        for sl in conditions:
            a, lo, hi = cell(name, sl)
            aucs.append(a)
            los.append(lo)
            his.append(hi)
        ax.errorbar(
            xs + offs, aucs, yerr=[los, his],
            color=PALETTE[i % len(PALETTE)], marker=_MARKERS[i % len(_MARKERS)],
            ms=8 if name.startswith("Ours") else 6, capsize=3, ls="none",
            elinewidth=1.4, label=name, zorder=5 if name.startswith("Ours") else 3,
        )
    ax.axhline(0.5, color="grey", ls=":", lw=1.0, label="chance")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Overall\n(pooled)", "Propofol", "Sevoflurane", "Volunteers"])
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_ylabel("Pooled out-of-fold AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(
        "Consciousness classification AUC across drugs & cohorts\n"
        "Bars: case-level BCa 95% CI   -   ours: 30-bin epochs; Abel: native 2-s bins",
        fontsize=11,
    )
    ax.legend(loc="lower left", frameon=True, fontsize=8.5, ncol=2)

    out = os.path.join(_IMG_DIR, out_name)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def _plot_slice(curves, res, model_names, slice_key, out_name):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for i, name in enumerate(model_names):
        yy, ss, gg, mm = curves[name]
        m = mm[slice_key]
        yo, so = yy[m], ss[m]
        ok = ~numpy.isnan(so)
        if len(numpy.unique(yo[ok])) < 2:
            continue
        fpr, tpr, _ = roc_curve(yo[ok], so[ok])
        r = res[(res.model == name) & (res.slice == slice_key)].iloc[0]
        ax.plot(
            fpr, tpr,
            color=PALETTE[i % len(PALETTE)],
            ls=LINE_STYLES[i % len(LINE_STYLES)],
            lw=2.6 if name.startswith("Ours") else 1.7,
            label=f"{name}  {r.auc:.3f}",
        )
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1.0, label="chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC - {SLICE_LABEL[slice_key]} (pooled OOF)")
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    out = os.path.join(_IMG_DIR, out_name)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def _plot(curves, res, model_names):
    _plot_auc_by_cohort(res, model_names, "abel_comparison.svg")
    _plot_slice(curves, res, model_names, "overall", "abel_comparison_overall.svg")
    _plot_slice(curves, res, model_names, "propofol", "abel_comparison_propofol.svg")
    _plot_slice(curves, res, model_names, "volunteers", "abel_comparison_volunteers.svg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=60)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-hmm", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_iter = min(args.n_iter, 10)
        args.n_boot = min(args.n_boot, 500)

    if args.replot:
        curves, model_names = _load_curves()
        res = pd.read_csv(os.path.join(_OUT_DIR, "abel_comparison_auc.csv"))
        _plot(curves, res, model_names)
    else:
        with A.time_block("abel_comparison_total"):
            run_comparison(args)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
