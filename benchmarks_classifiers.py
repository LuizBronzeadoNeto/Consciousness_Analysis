import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from analysis import load_data, FEATURE_COLS, _auc_with_ci, time_block

def get_search_spaces():
    """Define os pipelines e espaços de busca hiperparamétrica para cada modelo."""
    inner_cv = GroupKFold(n_splits=5)
    
    # 1. SVM-RBF (Modelo Original)
    pipe_svm = make_pipeline(
        PowerTransformer(method="yeo-johnson", standardize=True),
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    )
    param_svm = {
        "svc__C": np.logspace(-1, 3, 50),
        "svc__gamma": np.logspace(-4, 0, 50),
    }

    # 2. Random Forest
    pipe_rf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    )
    param_rf = {
        "randomforestclassifier__n_estimators": [100, 200, 300],
        "randomforestclassifier__max_depth": [5, 10, 15, None],
        "randomforestclassifier__min_samples_split": [2, 5, 10],
        "randomforestclassifier__max_features": ["sqrt", "log2", None]
    }

    # 3. XGBoost
    pipe_xgb = make_pipeline(
        StandardScaler(),
        XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    )
    param_xgb = {
        "xgbclassifier__n_estimators": [100, 200, 300],
        "xgbclassifier__max_depth": [3, 6, 9],
        "xgbclassifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "xgbclassifier__subsample": [0.7, 0.8, 1.0],
        "xgbclassifier__colsample_bytree": [0.7, 0.8, 1.0],
    }

    # 4. MLP (Rede Neural Rasa)
    pipe_mlp = make_pipeline(
        PowerTransformer(method="yeo-johnson", standardize=True),
        MLPClassifier(max_iter=500, early_stopping=True, random_state=42)
    )
    param_mlp = {
        "mlpclassifier__hidden_layer_sizes": [(32, 16), (64, 32), (32,)],
        "mlpclassifier__activation": ["relu", "tanh"],
        "mlpclassifier__alpha": [1e-4, 1e-3, 1e-2],
        "mlpclassifier__learning_rate_init": [1e-3, 1e-2]
    }

    models = {
        "SVM-RBF": (pipe_svm, param_svm),
        "Random Forest": (pipe_rf, param_rf),
        "XGBoost": (pipe_xgb, param_xgb),
        "MLP (Light NN)": (pipe_mlp, param_mlp)
    }
    
    return models, inner_cv

def run_benchmark():
    print("=== Iniciando Benchmark de Algoritmos sobre o Mesmos Atributos Híbridos ===")
    X, y, df, groups = load_data()
    
    propofol_mask = (df["compound"] == "pure_propofol").values
    sevoflurane_mask = df["compound"].isin(["mixed", "pure_sevo"]).values
    volunteer_mask = (df["cohort"] != "OR").values

    models_dict, inner_cv = get_search_spaces()
    cv_outer = GroupKFold(n_splits=5)
    
    oof_predictions = {name: np.zeros(len(y)) for name in models_dict}
    fold_aucs = {name: [] for name in models_dict}

    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y, groups)):
        print(f"\n--- Processando Fold {fold + 1}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        for name, (pipeline, search_space) in models_dict.items():
            with time_block(f"train_{name}_fold_{fold}"):
                search = RandomizedSearchCV(
                    pipeline,
                    search_space,
                    n_iter=20, ## <--
                    cv=inner_cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                    random_state=42 + fold
                )
                search.fit(X_train, y_train, groups=groups_train)
                best_model = search.best_estimator_

                if hasattr(best_model, "predict_proba"):
                    preds = best_model.predict_proba(X_test)[:, 1]
                else:
                    preds = best_model.decision_function(X_test)

                oof_predictions[name][test_idx] = preds
                fold_auc = roc_auc_score(y_test, preds)
                fold_aucs[name].append(fold_auc)
                print(f"[{name}] Fold {fold + 1} AUC: {fold_auc:.4f}")

    print("\n" + "="*60)
    print("=== RESULTADOS COMPARATIVOS (POOLED OUT-OF-FOLD) ===")
    print("="*60)

    summary_rows = []
    
    for name in models_dict:
        oof = oof_predictions[name]
        
        # 1. AUC Geral (Window-level)
        auc_gen, lo_gen, hi_gen = _auc_with_ci(y, oof, groups)
        
        # 2. AUC Propofol
        auc_prop, _, _ = _auc_with_ci(y[propofol_mask], oof[propofol_mask], groups[propofol_mask])
        
        # 3. AUC Sevoflurano
        auc_sevo, _, _ = _auc_with_ci(y[sevoflurane_mask], oof[sevoflurane_mask], groups[sevoflurane_mask])
        
        # 4. AUC Voluntários
        auc_vol, _, _ = _auc_with_ci(y[volunteer_mask], oof[volunteer_mask], groups[volunteer_mask])

        # 5. AUC Agregada por Caso/Estado
        df_agg = (
            df.assign(score=oof)
            .groupby(["case", "state"], as_index=False)
            .agg(score=("score", "mean"), label=("label", "first"))
        )
        auc_case = roc_auc_score(df_agg["label"], df_agg["score"])

        summary_rows.append({
            "Algoritmo": name,
            "AUC Geral (IC 95%)": f"{auc_gen:.4f} [{lo_gen:.3f}-{hi_gen:.3f}]",
            "AUC Propofol": f"{auc_prop:.4f}",
            "AUC Sevo": f"{auc_sevo:.4f}",
            "AUC Voluntários": f"{auc_vol:.4f}",
            "AUC por Caso (Agg)": f"{auc_case:.4f}"
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Teste de Significância Estatística (Wilcoxon entre SVM-RBF e os outros)
    print("\n" + "="*60)
    print("=== TESTE ESTATÍSTICO DE COMPARABILIDADE (vs SVM-RBF) ===")
    print("="*60)
    svm_oof = oof_predictions["SVM-RBF"]
    
    for name in models_dict:
        if name == "SVM-RBF":
            continue
        other_oof = oof_predictions[name]
        stat, p_val = wilcoxon(svm_oof, other_oof)
        diff = np.mean(svm_oof) - np.mean(other_oof)
        print(f"SVM-RBF vs {name}: Wilcoxon p-value = {p_val:.5e} | Diferença Média Predita = {diff:.4f}")

if __name__ == "__main__":
    run_benchmark()