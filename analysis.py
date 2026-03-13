import numpy
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
import complexity_calculations as eeg
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


OUTLIER_THRESHOLD = 35


def load_data():
    files = glob.glob("dataset/**/*_Sdb.csv", recursive=True)

    results = []
    print(
        f"Found {len(files)} files. Processing by State (Conscious vs Unconscious)..."
    )

    for filepath in files:
        fname = os.path.basename(filepath)
        case_id = fname.split("_")[0]
        cohort = os.path.basename(os.path.dirname(filepath))

        base_path = os.path.dirname(filepath)

        try:
            sdb = pd.read_csv(filepath, header=None).values
            f = pd.read_csv(
                os.path.join(base_path, f"{case_id}_f.csv"), header=None
            ).values.flatten()

            l_path = os.path.join(base_path, f"{case_id}_l.csv")
            if not os.path.exists(l_path):
                continue
            labels = pd.read_csv(l_path, header=None).values.flatten()

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
                    q = pd.read_csv(q_path, header=None).values.flatten()
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

                if len(alpha_state) < 15:
                    continue

                if numpy.std(alpha_state) == 0:
                    continue

                LZ = eeg.lempel_ziv_complexity(alpha_state)

                alpha_norm = (alpha_state - numpy.mean(alpha_state)) / numpy.std(
                    alpha_state
                )
                K = eeg.median_K(alpha_norm)

                delta_state = delta[state_mask]
                delta_alpha_ratio = numpy.mean(delta_state) / (
                    numpy.mean(alpha_state) + 1e-10
                )

                results.append(
                    {
                        "case": case_id,
                        "state": state_name,
                        "K": K,
                        "LZ": LZ,
                        "delta_alpha_ratio": delta_alpha_ratio,
                        "n_samples": len(alpha_state),
                    }
                )

        except Exception as e:
            print(f"Error processing {case_id}: {e}")

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

    param_grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": [1, 0.1, 0.01, 0.001, "scale"],
        "svc__kernel": ["rbf"],
    }

    pipeline_svm = make_pipeline(
        RobustScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True)
    )

    grid_search = GridSearchCV(
        pipeline_svm, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1
    ).fit(X, y, groups=groups)

    print(f"[SVM] Melhores parâmetros: {grid_search.best_params_}")

    return grid_search.best_estimator_


def main():
    X, y, df, groups = load_data()
    cv = GroupKFold(n_splits=5)

    print(f"Total de amostras (linhas): {len(df)}")
    print(f"Total de colunas (features): {X.shape[1]}")
    print(f"Pacientes únicos: {len(numpy.unique(groups))}")
    print(f"Conscious: {(y == 1).sum()}, Unconscious: {(y == 0).sum()}")

    svm_tuned = tuning_svm(X, y, groups)

    models = {
        "Logistic Regression (Quadratic)": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2),
            LogisticRegression(C=1.0, class_weight="balanced"),
        ),
        "SVM (RBF Kernel)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", class_weight="balanced")
        ),
        "SVM (Grid Search/Optimized)": svm_tuned,
    }

    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, groups=groups)
        print(
            f"\n{name}: 3D CV Accuracy = {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})"
        )

    fig, axes = plt.subplots(
        1, len(models), figsize=(8 * len(models), 7), subplot_kw={"projection": "3d"}
    )

    for ax, (name, model) in zip(axes, models.items()):
        cv_scores = cross_val_score(model, X, y, cv=cv, groups=groups)
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
        kk, ll = numpy.meshgrid(k_range, lz_range)

        delta_range = numpy.linspace(
            X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5, 30
        )
        surface_k, surface_lz, surface_delta = [], [], []
        for ki in k_range:
            for li in lz_range:
                grid_points = numpy.column_stack(
                    [
                        numpy.full(len(delta_range), ki),
                        numpy.full(len(delta_range), li),
                        delta_range,
                    ]
                )
                grid_unscaled = scaler.inverse_transform(grid_points)
                preds = model.predict(grid_unscaled)
                changes = numpy.where(numpy.diff(preds))[0]
                if len(changes) > 0:
                    idx = changes[0]
                    boundary_point = scaler.inverse_transform(
                        grid_points[idx : idx + 1]
                    )[0]
                    surface_k.append(boundary_point[0])
                    surface_lz.append(boundary_point[1])
                    surface_delta.append(boundary_point[2])

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
            f"{name}\n3D CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})"
        )
        ax.legend(fontsize="x-small", loc="upper left")

    plt.suptitle("3D Classification (K, LZ, Delta/Alpha Ratio)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
