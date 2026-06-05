import argparse
import os

import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut

import abel_comparison as AC
import analysis as A


def _ours_loo(X8, y, groups, test_cases, restrict_train, n_iter, seed):
    oof = numpy.full(len(y), numpy.nan)
    for held in test_cases:
        te = numpy.flatnonzero(groups == held)
        tr = numpy.flatnonzero(groups != held)
        if restrict_train is not None:
            tr = tr[numpy.isin(groups[tr], list(restrict_train))]
        s, _ = AC._tuned_oof(
            AC._make_ours, AC._OURS_DIST, X8, y, groups, tr, te, n_iter, seed, proba=False
        )
        oof[te] = s
    return oof


def run(args):
    X8, y, df, groups = A.load_data()
    df = df.reset_index(drop=True)
    X8 = df[A.FEATURE_COLS].values
    y = df["label"].values.astype(int)
    groups = df["case"].values
    vol_mask = (df["cohort"] != "OR").values
    vol_cases = list(numpy.unique(groups[vol_mask]))
    print(f"\n{len(vol_cases)} volunteers, {int(vol_mask.sum())} epochs")

    oof_all = _ours_loo(X8, y, groups, vol_cases, None, args.n_iter, args.seed)
    oof_vol = _ours_loo(X8, y, groups, vol_cases, set(vol_cases), args.n_iter, args.seed)
    yv = y[vol_mask]
    auc_all = roc_auc_score(yv, oof_all[vol_mask])
    auc_vol = roc_auc_score(yv, oof_vol[vol_mask])
    if auc_all >= auc_vol:
        ours_oof, ours_auc, regime = oof_all[vol_mask], auc_all, "trained on all other cases"
    else:
        ours_oof, ours_auc, regime = oof_vol[vol_mask], auc_vol, "trained on volunteers only"
    print(f"ours: all-data {auc_all:.3f} vs vol-only {auc_vol:.3f}  -> keeping {ours_auc:.3f}")

    vol_df = df[vol_mask].reset_index(drop=True)
    Xb, yb, bins = AC.load_perbin(vol_df)
    gb = bins["case"].values
    print(f"abel: {len(yb)} bins over {len(numpy.unique(gb))} subjects")

    abel = {n: numpy.full(len(yb), numpy.nan) for n in ("Abel Sdb", "Abel PCA", "Abel LDA+HMM2")}
    for tr, te in LeaveOneGroupOut().split(Xb, yb, gb):
        abel["Abel Sdb"][te] = AC._make_abel_sdb().fit(Xb[tr], yb[tr]).predict_proba(Xb[te])[:, 1]
        abel["Abel PCA"][te] = AC._make_abel_pca().fit(Xb[tr], yb[tr]).predict_proba(Xb[te])[:, 1]
        s, _ = AC._lda_hmm2_perbin(Xb, yb, bins, tr, te)
        m = ~numpy.isnan(s)
        abel["Abel LDA+HMM2"][m] = s[m]
    abel_auc = {k: roc_auc_score(yb, v) for k, v in abel.items()}

    print(f"\nours {ours_auc:.3f}  ({regime})")
    for k in ("Abel Sdb", "Abel PCA", "Abel LDA+HMM2"):
        print(f"{k} {abel_auc[k]:.3f}")

    _plot(yv, ours_oof, ours_auc, regime, yb, abel, abel_auc)


def _plot(yv, ours_oof, ours_auc, regime, yb, abel, abel_auc):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    series = [("Ours (8-feat SVM)", yv, ours_oof, ours_auc)]
    for name in ("Abel Sdb", "Abel PCA", "Abel LDA+HMM2"):
        series.append((name, yb, abel[name], abel_auc[name]))

    for i, (name, yy, ss, auc) in enumerate(series):
        ok = ~numpy.isnan(ss)
        fpr, tpr, _ = roc_curve(yy[ok], ss[ok])
        ax.plot(
            fpr, tpr,
            color=AC.PALETTE[i % len(AC.PALETTE)],
            ls=AC.LINE_STYLES[i % len(AC.LINE_STYLES)],
            lw=2.6 if i == 0 else 1.7,
            label=f"{name}  {auc:.3f}",
        )
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1.0, label="chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        "ROC - Volunteers, leave-one-subject-out (each model at its prime)\n"
        f"Abel: volunteer-only, per-bin   |   Ours: {regime}, epochs",
        fontsize=10,
    )
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    out = os.path.join(AC._IMG_DIR, "abel_comparison_prime_volunteers.svg")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_iter = min(args.n_iter, 10)

    with A.time_block("volunteer_prime_total"):
        run(args)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
