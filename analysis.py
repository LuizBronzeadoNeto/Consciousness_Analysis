import numpy
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
import complexity_calculations as eeg


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
                    "LZ": LZ
                })

        except Exception as e:
            print(f"Error processing {case_id}: {e}")
    
    df = pd.DataFrame(results)
    # Map categories: Conscious=1, Unconscious=0
    df['label'] = df['state'].map({'Conscious': 1, 'Unconscious': 0})

    X = df[['K', 'LZ']].values  
    y = df['label'].values

    # 2. Create the "Elliptical" Model Pipeline
    # Degree 2 = Allows parabolas and ellipses (Quadratic)
    model = make_pipeline(
        StandardScaler(),            # Step 1: Normalize data (Helps curve fitting)
        PolynomialFeatures(degree=2), # Step 2: Create squares (The "Curve" magic)
        LogisticRegression(C=1.0)     # Step 3: Classify
    )

    model.fit(X, y)

    # 3. Plotting
    plt.figure(figsize=(10, 6))

    # A. Draw the Curved Decision Boundary
    # This will color the background based on the curve
    DecisionBoundaryDisplay.from_estimator(
        model, X, 
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.2, 
        cmap="coolwarm", # Blue=Conscious zone, Red=Unconscious zone
        ax=plt.gca(),
        response_method="predict"
    )

    # B. Plot the actual data points
    conscious = df[df['label']==1]
    unconscious = df[df['label']==0]

    plt.scatter(conscious['K'], conscious['LZ'], 
                color='red', label='Conscious', edgecolor='k', s=50)
    plt.scatter(unconscious['K'], unconscious['LZ'], 
                color='blue', label='Unconscious', marker='x', s=50)

    plt.xlabel("Median K (Chaos)")
    plt.ylabel("Lempel-Ziv Complexity")
    plt.title(f"Quadratic Classification (Elliptical Boundary)\nAccuracy: {model.score(X, y):.2%}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()