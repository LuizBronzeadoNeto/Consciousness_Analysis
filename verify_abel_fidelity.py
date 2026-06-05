import os

import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import abel_comparison as AC
import analysis as A

VOL_DIR = "Volunteer"
TRAIN_CASES = ["02", "03", "04", "05", "07", "08", "09"]
ABEL_PUBLISHED = {"Sdb": 0.9568, "PCA": 0.9581, "LDA+HMM2": 0.9704}


def load_case(case):
    base = os.path.join("dataset", VOL_DIR)
    sdb = A._load_csv_cached(os.path.join(base, f"{case}_Sdb.csv"), "Sdb", case, VOL_DIR, [])
    f = A._load_csv_cached(os.path.join(base, f"{case}_f.csv"), "f", case, VOL_DIR, []).flatten()
    labels = A._load_csv_cached(os.path.join(base, f"{case}_l.csv"), "l", case, VOL_DIR, []).flatten()
    t = A._load_csv_cached(os.path.join(base, f"{case}_t.csv"), "t", case, VOL_DIR, []).flatten()
    spec = sdb.T if sdb.shape[0] == len(f) else sdb
    n = min(spec.shape[0], len(labels), len(t))
    spec, labels, t = spec[:n], labels[:n], t[:n]
    keep = numpy.isin(labels, [0, 1])
    return spec[keep], labels[keep].astype(int), t[keep]


DATA = {c: load_case(c) for c in TRAIN_CASES}


def _case_lengths(t):
    dt = numpy.median(numpy.diff(t)) if len(t) > 1 else 2.0
    lengths, run = [], 1
    for a, b in zip(t[:-1], t[1:]):
        if b - a <= 1.5 * dt:
            run += 1
        else:
            lengths.append(run)
            run = 1
    lengths.append(run)
    return lengths


def _lda_hmm2(train_cases, held):
    from hmmlearn.hmm import GaussianHMM

    Xtr = numpy.vstack([DATA[c][0] for c in train_cases])
    ytr = numpy.concatenate([DATA[c][1] for c in train_cases])
    lda = LinearDiscriminantAnalysis(n_components=1).fit(Xtr, ytr)

    ld_parts, lengths = [], []
    for c in train_cases:
        Xc, _, tc = DATA[c]
        ld_parts.append(lda.transform(Xc)[:, 0])
        lengths += _case_lengths(tc)
    ld_tr = numpy.concatenate(ld_parts)

    hmm = GaussianHMM(
        n_components=2, covariance_type="diag", algorithm="viterbi",
        n_iter=10, random_state=42,
    )
    hmm.fit(ld_tr.reshape(-1, 1), lengths)
    post_tr = hmm.predict_proba(ld_tr.reshape(-1, 1), lengths)
    lr = LogisticRegression(max_iter=5000).fit(post_tr, ytr)

    Xte, _, tte = DATA[held]
    ld_te = lda.transform(Xte)[:, 0]
    post_te = hmm.predict_proba(ld_te.reshape(-1, 1), _case_lengths(tte))
    return lr.predict_proba(post_te)[:, 1]


def loo_aucs(kind):
    aucs = []
    for held in TRAIN_CASES:
        train_cases = [c for c in TRAIN_CASES if c != held]
        Xte, yte, _ = DATA[held]
        if kind == "Sdb":
            Xtr = numpy.vstack([DATA[c][0] for c in train_cases])
            ytr = numpy.concatenate([DATA[c][1] for c in train_cases])
            score = AC._make_abel_sdb().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        elif kind == "PCA":
            Xtr = numpy.vstack([DATA[c][0] for c in train_cases])
            ytr = numpy.concatenate([DATA[c][1] for c in train_cases])
            score = AC._make_abel_pca().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        else:
            score = _lda_hmm2(train_cases, held)
        aucs.append(roc_auc_score(yte, score))
    return aucs


def main():
    print("abel models on the volunteers (LOO), should land on his published means\n")
    for kind in ["Sdb", "PCA", "LDA+HMM2"]:
        aucs = loo_aucs(kind)
        m = float(numpy.mean(aucs))
        pub = ABEL_PUBLISHED[kind]
        tag = "ok" if abs(m - pub) < 0.03 else "off"
        folds = ", ".join(f"{a:.2f}" for a in aucs)
        print(f"  {kind:9} {m:.3f}  (abel {pub:.3f})  {tag}   [{folds}]")


if __name__ == "__main__":
    main()
