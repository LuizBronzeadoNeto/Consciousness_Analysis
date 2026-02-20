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

    model = make_pipeline(
        StandardScaler(),           
        PolynomialFeatures(degree=2), 
        LogisticRegression(C=1.0, class_weight='balanced')
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv)

    model.fit(X, y)

    plt.figure(figsize=(10, 6))


    DecisionBoundaryDisplay.from_estimator(
        model, X, 
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.2, 
        cmap="coolwarm",
        ax=plt.gca(),
        response_method="predict"
    )

    conscious = df[df['label']==1]
    unconscious = df[df['label']==0]

    ax = plt.gca()

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

    plt.xlabel("Median K (Chaos)")
    plt.ylabel("Lempel-Ziv Complexity")
    plt.title(f"Quadratic Classification (Elliptical Boundary)\n5-Fold CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()