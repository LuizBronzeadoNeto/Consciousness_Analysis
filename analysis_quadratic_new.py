import numpy
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import complexity_calculations as eeg
import seaborn as sns


OUTLIER_THRESHOLD = 35


def main():
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

                LZ = eeg.lempel_ziv_complexity(alpha_state)

                alpha_norm = (alpha_state - numpy.mean(alpha_state)) / numpy.std(alpha_state)
                K = eeg.median_K(alpha_norm)

                C = eeg.criticality_proximity(K, alpha=0.85)

                results.append({
                    "case": case_id,
                    "state": state_name,
                    "K": K,
                    "LZ": LZ,
                    "C": C,
                    "n_samples": len(alpha_state)
                })

        except Exception as e:
            print(f"Error processing {case_id}: {e}")
    
    df = pd.DataFrame(results)
    
    col_k = 'Median K (Chaos)'
    col_lz = 'Lempel-Ziv Complexity'
    col_c = 'Proximity to Criticality Measure (C)'
    
    df[[col_k, col_lz, col_c]] = df[['K', 'LZ', 'C']].astype(float)
    df['label'] = df['state'].map({'Conscious': 1, 'Unconscious': 0})

    X = df[['K', 'LZ', 'C']].values  
    y = df['label'].values

    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LogisticRegression(class_weight='balanced'))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv)
    model.fit(X, y)

    #fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    plt.figure(figsize=(10, 7))
    axes = [plt.gca()]
    projections = [
        (col_k, col_lz, "Median K (Chaos) x Lempel-Ziv Complexity"),
        #(col_k, col_c, "Median K (Chaos) x Proximity to Criticality (C)"),
        #(col_lz, col_c, "Lempel-Ziv Complexity x Proximity to Criticality (C)")
    ]

    for i, (feat_x, feat_y, title) in enumerate(projections):
        ax = axes[i]
        X_2d = df[[feat_x, feat_y]].values
        
        vis_model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LogisticRegression(class_weight='balanced'))
        vis_model.fit(X_2d, y)

        DecisionBoundaryDisplay.from_estimator(
            vis_model, X_2d, plot_method="pcolormesh", shading="auto",
            alpha=0.15, cmap="coolwarm", ax=ax, response_method="predict"
        )

        is_outlier = df['n_samples'] <= OUTLIER_THRESHOLD
        
        for state, color, marker in [('Conscious', 'red', 'o'), ('Unconscious', 'blue', 'x')]:
            subset = df[df['state'] == state]
            ok = subset[~is_outlier[subset.index]]
            short = subset[is_outlier[subset.index]]
            
            edge = 'k' if marker != 'x' else None
            
            ax.scatter(ok[feat_x], ok[feat_y], color=color, label=state, 
                       edgecolor=edge, s=60, marker=marker)
            
            if not short.empty:
                ax.plot(short[feat_x].values, short[feat_y].values, marker,
                        color=color, markerfacecoloralt='yellow', fillstyle='left',
                        markersize=8, markeredgecolor=edge if edge else color, 
                        linestyle='None',
                        label=f'{state} (n\u2264{OUTLIER_THRESHOLD})')

        ax.set_title(title)
        ax.set_xlabel(feat_x)
        ax.set_ylabel(feat_y)
        ax.legend(loc='upper left', fontsize='x-small')

    plt.suptitle(f"Quadratic Classification (Elliptical Boundary)\n5-Fold CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()