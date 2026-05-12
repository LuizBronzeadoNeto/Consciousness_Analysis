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
from sklearn.metrics import roc_auc_score
from scipy.stats import loguniform
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
    "LZ",
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
                LZ = eeg.lempel_ziv_complexity(alpha_win)
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
                        "LZ": LZ,
                        "log_ratio": log_ratio,
                        "log_alpha_var": log_alpha_var,
                        "log_theta": float(numpy.log(theta[sl].mean() + 1e-12)),
                        "log_beta": float(numpy.log(beta[sl].mean() + 1e-12)),
                        "log_gamma": float(numpy.log(gamma[sl].mean() + 1e-12)),
                        "spectral_entropy": spectral_entropy,
                        "n_samples": min_len,
                        "compound": compound,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use cuML GPU SVM (requires cuml installed; drops Nystroem, uses exact RBF)",
    )
    args = parser.parse_args()

    with time_block("total_run", gpu=args.gpu):
        X, y, df, groups = load_data()
        cv = GroupKFold(n_splits=5)
        propofol_mask = df["compound"] == "pure_propofol"
        sevoflurane_mask = df["compound"].isin(["mixed", "pure_sevo"])

        print(f"Total samples (lines): {len(df)}")
        print(f"Total rows (features): {X.shape[1]}")
        print(f"Unique patients: {len(numpy.unique(groups))}")
        print(f"Conscious: {(y == 1).sum()}, Unconscious: {(y == 0).sum()}")

        if args.gpu:
            X = numpy.ascontiguousarray(X, dtype=numpy.float32)
            y = numpy.ascontiguousarray(y, dtype=numpy.int32)

        svm_tuned = tuning_svm(X, y, groups, gpu=args.gpu)

        models = {
            "SVM (Randomized Search/Optimized)": svm_tuned,
        }

        auc_scores = {name: [] for name in models}

        y_score_full = svm_tuned.decision_function(X)

        auc_propofol = roc_auc_score(y[propofol_mask], y_score_full[propofol_mask])
        auc_sevo = roc_auc_score(y[sevoflurane_mask], y_score_full[sevoflurane_mask])
        print(f"SVM (Randomized Search/Optimized): median propofol AUC -> {auc_propofol}")
        print(f"SVM (Randomized Search/Optimized): median sevoflurane + mixed auc {auc_sevo}")

        oof_scores = {name: numpy.full(len(y), numpy.nan) for name in models}
        with time_block("auc_loop_total", n_folds=5, n_models=len(models)):
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                for name, model in models.items():
                    with time_block("cv_fold", fold=fold_idx, model=name):
                        model.fit(X_train, y_train)
                        if hasattr(model, "predict_proba"):
                            y_score = model.predict_proba(X_test)[:, 1]
                        elif hasattr(model, "decision_function"):
                            y_score = model.decision_function(X_test)
                        else:
                            raise ValueError(f"{name} has no scoring method")
                        auc = roc_auc_score(y_test, y_score)
                        auc_scores[name].append(auc)
                        oof_scores[name][test_idx] = y_score

        auc_results = {}
        for name, scores in auc_scores.items():
            print(f"\n{name}: window-level mean AUC = {numpy.mean(scores):.4f}")
            auc_results[name] = sum(scores) / len(scores)

            agg = (
                df.assign(score=oof_scores[name])
                .groupby(["case", "state"], as_index=False)
                .agg(score=("score", "mean"), label=("label", "first"))
            )
            print(
                f"{name}: case-(state) AUC = {roc_auc_score(agg.label, agg.score):.4f} "
                f"(n_pairs={len(agg)})"
            )

        # Compute CV accuracy once per model — previously computed twice
        # (here and again inside the plot loop).
        cv_scores_per_model = {}
        for name, model in models.items():
            with time_block("cv_accuracy", model=name):
                cv_scores_per_model[name] = cross_val_score(
                    model, X, y, cv=cv, groups=groups
                )
            scores = cv_scores_per_model[name]
            print(
                f"\n{name}: 3D CV Accuracy = {scores.mean():.2%} (+/- {scores.std():.2%})"
            )

        with time_block("plot_build"):
            fig, axes = plt.subplots(
                1,
                len(models),
                figsize=(8 * len(models), 7),
                subplot_kw={"projection": "3d"},
                squeeze=False,
            )
            axes = axes.ravel()

            for ax, (name, model) in zip(axes, models.items()):
                cv_scores = cv_scores_per_model[name]
                model.fit(X, y)

                is_outlier = df["n_samples"] <= OUTLIER_THRESHOLD

                for state, color, marker in [
                    ("Conscious", "red", "o"),
                    ("Unconscious", "blue", "x"),
                ]:
                    subset = df[df["state"] == state]
                    ok = subset[~is_outlier[subset.index]]
                    short = subset[is_outlier[subset.index]]

                    ax.scatter(
                        ok["K"],
                        ok["LZ"],
                        ok["log_ratio"],
                        color=color,
                        label=state,
                        marker=marker,
                        edgecolor="k" if marker == "o" else color,
                        s=50,
                    )
                    if not short.empty:
                        ax.scatter(
                            short["K"],
                            short["LZ"],
                            short["log_ratio"],
                            color="yellow",
                            label=f"{state} (n\u2264{OUTLIER_THRESHOLD})",
                            marker=marker,
                            edgecolor=color,
                            s=50,
                        )

                ax.set_xlabel("Median K (Chaos)")
                ax.set_ylabel("Lempel-Ziv Complexity")
                ax.set_zlabel("log(Delta/Alpha)")
                ax.set_title(
                    f"{name}\n3D CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})\nMean AUC: {auc_results[name]:.4f}"
                )
                ax.legend(fontsize="x-small", loc="upper left")

            plt.suptitle("3D Classification (K, LZ, log(Delta/Alpha))", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


if __name__ == "__main__":
    main()
