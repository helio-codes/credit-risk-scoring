import joblib
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from data_pipeline import load_and_preprocess

def run_experiment(data_path: str):
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(
        filepath=data_path,
        target_col='default'
    )
    
    # Modelo lidando com dados desbalanceados via SMOTE
    clf = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)),
        ('model', XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    print("--- Relatório de Classificação ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    joblib.dump(clf, "credit_risk_model.joblib")
    print("\nModelo serializado salvo com sucesso em credit_risk_model.joblib")

if __name__ == '__main__':
    run_experiment("data/raw/credit_data.csv")