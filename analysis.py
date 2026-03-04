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
from sklearn.model_selection import cross_val_score, StratifiedKFold
import complexity_calculations as eeg
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import savgol_filter


OUTLIER_THRESHOLD = 35


def load_data():
    files = glob.glob("dataset/**/*_Sdb.csv", recursive=True)

    results = []
    print(f"Found {len(files)} files. Processing by State (Conscious vs Unconscious)...")

    for filepath in files:
        fname = os.path.basename(filepath)
        case_id = fname.split("_")[0]
        cohort = os.path.basename(os.path.dirname(filepath))

        base_path = os.path.dirname(filepath)

        try:
            sdb = pd.read_csv(filepath, header=None).values
            f = pd.read_csv(os.path.join(base_path, f"{case_id}_f.csv"), header=None).values.flatten()

            l_path = os.path.join(base_path, f"{case_id}_l.csv")
            if not os.path.exists(l_path):
                continue
            labels = pd.read_csv(l_path, header=None).values.flatten()

            P = 10**(sdb/10)
            if P.shape[0] == len(f):
                alpha = P[(f >= 8) & (f <= 12), :].mean(axis=0)
            else:
                alpha = P[:, (f >= 8) & (f <= 12)].mean(axis=1)

            min_len = min(len(alpha), len(labels))

            quality_mask = numpy.ones(min_len, dtype=bool)
            if cohort == "OR":
                q_path = os.path.join(base_path, f"{case_id}_EEGquality.csv")
                if os.path.exists(q_path):
                    q = pd.read_csv(q_path, header=None).values.flatten()
                    min_len = min(min_len, len(q))
                    quality_mask = q[:min_len].astype(bool)

            alpha = alpha[:min_len]
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

                alpha_smooth = savgol_filter(alpha_state, window_length=11, polyorder=2)

                LZ = eeg.lempel_ziv_complexity(alpha_smooth)

                alpha_norm = (alpha_smooth - numpy.mean(alpha_smooth)) / numpy.std(alpha_smooth)
                K = eeg.median_K(alpha_norm)

                results.append({
                    "case": case_id,
                    "state": state_name,
                    "K": K,
                    "LZ": LZ,
                    "n_samples": len(alpha_state)
                })

        except Exception as e:
            print(f"Error processing {case_id}: {e}")

    df = pd.DataFrame(results)
    df['label'] = df['state'].map({'Conscious': 1, 'Unconscious': 0})

    X = df[['K', 'LZ']].values
    y = df['label'].values

    return X, y, df


def plot_scatter(ax, df):
    conscious = df[df['label'] == 1]
    unconscious = df[df['label'] == 0]

    is_outlier = (df['n_samples'] <= OUTLIER_THRESHOLD)

    c_ok = conscious[~is_outlier[conscious.index]]
    c_short = conscious[is_outlier[conscious.index]]
    u_ok = unconscious[~is_outlier[unconscious.index]]
    u_short = unconscious[is_outlier[unconscious.index]]

    ax.scatter(c_ok['K'], c_ok['LZ'],
               color='red', label='Conscious', edgecolor='k', s=50)
    ax.scatter(u_ok['K'], u_ok['LZ'],
               color='blue', label='Unconscious', marker='x', s=50)

    if len(c_short) > 0:
        ax.plot(c_short['K'].values, c_short['LZ'].values, 'o',
                color='red', markerfacecoloralt='yellow', fillstyle='left',
                markersize=7, markeredgecolor='k',
                label=f'Conscious (n\u2264{OUTLIER_THRESHOLD})', linestyle='None')
    if len(u_short) > 0:
        ax.plot(u_short['K'].values, u_short['LZ'].values, 's',
                color='blue', markerfacecoloralt='yellow', fillstyle='left',
                markersize=7, markeredgecolor='k',
                label=f'Unconscious (n\u2264{OUTLIER_THRESHOLD})', linestyle='None')

def tuning_svm(X, y, cv):

    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': [1, 0.1, 0.01, 0.001, 'scale'],
        'svc__kernel': ['rbf'] 
    }

    pipeline_svm = make_pipeline(
        RobustScaler(), 
        SVC(kernel='rbf', class_weight='balanced', probability=True)
    )

    grid_search = GridSearchCV(
        pipeline_svm, 
        param_grid, 
        cv=cv, 
        scoring='accuracy', 
        n_jobs=-1
    ).fit(X, y)

    print(f"[SVM] Melhores parâmetros: {grid_search.best_params_}")

    return grid_search.best_estimator_

def main():
    X, y, df = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Total de amostras (linhas): {len(df)}")
    print(f"Total de colunas (features): {X.shape[1]}")

    svm_tuned = tuning_svm(X, y, cv)

    models = {
        "Logistic Regression (Quadratic)": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2),
            LogisticRegression(C=1.0, class_weight='balanced')
        ),
        "SVM (RBF Kernel)": make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', class_weight='balanced')
        ),
        "SVM (Grid Search/Optimized)": svm_tuned,
    }

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    for i, (name, model) in enumerate(models.items()):
            cv_scores = cross_val_score(model, X, y, cv=cv)
            model.fit(X, y)
            y_pred = model.predict(X)

            ax_boundary = axes[0, i]
            DecisionBoundaryDisplay.from_estimator(
                model, X,
                plot_method="pcolormesh",
                shading="auto",
                alpha=0.2,
                cmap="coolwarm",
                ax=ax_boundary,
                response_method="predict"
            )
            plot_scatter(ax_boundary, df)
            ax_boundary.set_xlabel("Median K (Chaos)")
            ax_boundary.set_ylabel("LZ Complexity")
            ax_boundary.set_title(f"{name}\nCV Acc: {cv_scores.mean():.2%}")
            ax_boundary.legend(fontsize='x-small')

            ax_cm = axes[1, i]
            cm = confusion_matrix(y, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Uncon.', 'Cons.'])
            disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
            ax_cm.set_title(f"Confusion Matrix: {name}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
