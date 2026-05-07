import numpy
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GroupKFold
import complexity_calculations as eeg
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from joblib import Parallel, delayed
import logging
import json
import time
import contextlib
import datetime
import sys
import yaml
from sklearn.kernel_approximation import Nystroem
OUTLIER_THRESHOLD = 35
WINDOW_SIZE = 3
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
        if (
            os.path.exists(npy_path)
            and os.path.getmtime(npy_path) >= os.path.getmtime(csv_path)
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

def identify_compound(case_id):
    with open("dataset/OR/rx_sorted_case_ids.yml", 'r') as file:
        rx_sorted_case_ids = yaml.safe_load(file)
        compound = next(k for k, v in rx_sorted_case_ids.items() if case_id in v)
        if compound == None: raise ValueError(f"invalid case id: {case_id}")
    return compound


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
            "f", case_id, cohort, perf_records,
        ).flatten()

        l_path = os.path.join(base_path, f"{case_id}_l.csv")
        if not os.path.exists(l_path):
            return results, perf_records
        labels = _load_csv_cached(l_path, "l", case_id, cohort, perf_records).flatten()

        P = 10 ** (sdb / 10)
        if P.shape[0] == len(f):
            alpha = P[(f >= 8) & (f <= 12), :].mean(axis=0)
            delta = P[(f >= 1) & (f <= 4), :].mean(axis=0)
        else:
            alpha = P[:, (f >= 8) & (f <= 12)].mean(axis=1)
            delta = P[:, (f >= 1) & (f <= 4)].mean(axis=1)

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
        labels = labels[:min_len]
        quality_mask = quality_mask[:min_len]

        states = {0: "Unconscious", 1: "Conscious"}
        for state_val, state_name in states.items():
            state_mask = (labels == state_val) & quality_mask
            alpha_state = alpha[state_mask]
            delta_state = delta[state_mask]
            
            if numpy.std(alpha_state) == 0:
                continue

            alpha_norm = (alpha_state - numpy.mean(alpha_state)) / numpy.std(
                alpha_state
            )

            compound = ''
            if cohort == 'OR':
                compound = identify_compound(case_id)
            else: compound = 'pure_propofol'
            for epoch, i in enumerate(range(0, len(alpha_state) - WINDOW_SIZE, WINDOW_SIZE)):
                with _collect_block(perf_records, "epoch_metrics", case=case_id, state=state_name, epoch_id=epoch):
                    alpha_group = alpha_state[i:i + WINDOW_SIZE]
                    delta_group = delta_state[i:i + WINDOW_SIZE]
                    LZ = eeg.lempel_ziv_complexity(alpha_group)
                    K = eeg.median_K(alpha_group)
                    delta_alpha_ratio = numpy.mean(delta_group) / (
                        numpy.mean(alpha_group) + 1e-10
                    )
                    results.append(
                        {
                            "case": case_id,
                            "state": state_name,
                            "epoch_number": epoch,
                            "K": K,
                            "LZ": LZ,
                            "delta_alpha_ratio": delta_alpha_ratio,
                            "n_samples": len(alpha_state),
                            "compound": compound,
                        }
                    )
            unprocessed_samples = len(alpha_state) % 3
            if unprocessed_samples > 0:
                LZ = eeg.lempel_ziv_complexity(alpha_state[-WINDOW_SIZE:])
                K = eeg.median_K(alpha_norm[-WINDOW_SIZE:])
                results.append(
                    {
                        "case": case_id,
                        "state": state_name,
                        "K": K,
                        "LZ": LZ,
                        "delta_alpha_ratio": delta_alpha_ratio,
                        "n_samples": len(alpha_state),
                        "compound": compound,
                    }
                )
          

    except Exception as e:
        # Surface the failure in perf log so it isn't silent
        print(f"ERROR{e}")
        perf_records.append(
            _make_record(
                "case_error", time.perf_counter() - case_t0,
                {"case": case_id, "error": repr(e)},
            )
        )

    perf_records.append(
        _make_record(
            "case", time.perf_counter() - case_t0,
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

        X = df[["K", "LZ", "delta_alpha_ratio"]].values
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


def tuning_svm(X, y, groups):
    inner_cv = GroupKFold(n_splits=5)

    param_dist = {
        "linearsvc__C": loguniform(0.1, 1000),
        #"nystroem__gamma": loguniform(1e-4, 1),
        #"svc__kernel": ["rbf"], 
    }

    pipeline_svm = make_pipeline(
        RobustScaler(),
        Nystroem(n_components=500, random_state=42),
        LinearSVC(class_weight="balanced", dual=False, max_iter=2000)
    )

    with time_block("tuning_svm", n_iter=20):
        search = RandomizedSearchCV(
            pipeline_svm, param_dist, n_iter=20, cv=inner_cv, scoring="roc_auc",
            n_jobs=-1, random_state=42,
        ).fit(X, y, groups=groups)

    print(f"[SVM] Best parameters:: {search.best_params_}")

    return search.best_estimator_


def main():
    with time_block("total_run"):
        X, y, df, groups = load_data()
        cv = GroupKFold(n_splits=5)
        propofol_mask = df["compound"] == "pure_propofol"
        sevoflurane_mask = df["compound"].isin(["mixed", "pure_sevo"])

        print(df[sevoflurane_mask][["case", "state", "n_samples"]].to_string())

        print(f"Total samples (lines): {len(df)}")
        print(f"Total rows (features): {X.shape[1]}")
        print(f"Unique patients: {len(numpy.unique(groups))}")
        print(f"Conscious: {(y == 1).sum()}, Unconscious: {(y == 0).sum()}")

        svm_tuned = tuning_svm(X, y, groups)

        models = {
            #"Logistic Regression (Quadratic)": make_pipeline(
             #   StandardScaler(),
              #  PolynomialFeatures(degree=2),
               # LogisticRegression(C=1.0, class_weight="balanced"),
            #),
            #"SVM (RBF Kernel)": make_pipeline(
             #   StandardScaler(), SVC(kernel="rbf", class_weight="balanced")
            #),
            "SVM (Randomized Search/Optimized)": svm_tuned,
        }

        auc_scores = {name: [] for name in models}

        #y_score_full = svm_tuned.decision_function(X)
        
        #auc_propofol = roc_auc_score(y[propofol_mask], y_score_full[propofol_mask])
        #auc_sevo = roc_auc_score(y[sevoflurane_mask], y_score_full[sevoflurane_mask])
        #print(f"svm propofol AUC {auc_propofol}")
        #print(f"svm sevo auc {auc_sevo}")

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

        auc_results = {}
        for name, scores in auc_scores.items():
            print(f"\n{name}: mean AUC = {numpy.mean(scores):.4f}")
            auc_results[name] = sum(scores) / len(scores)

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
                1, len(models), figsize=(8 * len(models), 7),
                subplot_kw={"projection": "3d"},
            )

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
                        ok["delta_alpha_ratio"],
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
                            short["delta_alpha_ratio"],
                            color="yellow",
                            label=f"{state} (n\u2264{OUTLIER_THRESHOLD})",
                            marker=marker,
                            edgecolor=color,
                            s=50,
                        )

                scaler = StandardScaler().fit(X)
                X_scaled = scaler.transform(X)
                k_range = numpy.linspace(
                    X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 30
                )
                lz_range = numpy.linspace(
                    X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 30
                )
                delta_range = numpy.linspace(
                    X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5, 30
                )

                # One batched predict over the full 30x30x30 grid replaces
                # 900 scalar `model.predict` calls. Same first-sign-change
                # semantics along the delta axis as the original triple loop.
                with time_block("boundary_grid", model=name):
                    KK, LL, DD = numpy.meshgrid(
                        k_range, lz_range, delta_range, indexing="ij"
                    )
                    grid_scaled = numpy.column_stack(
                        [KK.ravel(), LL.ravel(), DD.ravel()]
                    )
                    grid_unscaled = scaler.inverse_transform(grid_scaled)
                    preds = model.predict(grid_unscaled).reshape(30, 30, 30)
                    diffs = numpy.diff(preds, axis=2) != 0
                    has_change = diffs.any(axis=2)
                    first_idx = diffs.argmax(axis=2)
                    boundary = grid_unscaled.reshape(30, 30, 30, 3)
                    ki_arr, li_arr = numpy.where(has_change)
                    pts = boundary[ki_arr, li_arr, first_idx[has_change]]
                    surface_k = pts[:, 0].tolist()
                    surface_lz = pts[:, 1].tolist()
                    surface_delta = pts[:, 2].tolist()

                if surface_k:
                    from matplotlib.tri import Triangulation

                    try:
                        tri = Triangulation(surface_k, surface_lz)
                        ax.plot_trisurf(tri, surface_delta, alpha=0.25, color="purple")
                    except Exception:
                        ax.scatter(
                            surface_k,
                            surface_lz,
                            surface_delta,
                            color="purple",
                            alpha=0.1,
                            s=5,
                            label="Decision boundary",
                        )

                ax.set_xlabel("Median K (Chaos)")
                ax.set_ylabel("Lempel-Ziv Complexity")
                ax.set_zlabel("Delta/Alpha Power Ratio")
                ax.set_title(
                    f"{name}\n3D CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})\nMean AUC: {auc_results[name]:.4f}"
                )
                ax.legend(fontsize="x-small", loc="upper left")

            plt.suptitle("3D Classification (K, LZ, Delta/Alpha Ratio)", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


if __name__ == "__main__":
    main()
