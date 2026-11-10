import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_and_preprocess(filepath: str, target_col: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(filepath)
    
    # Separação entre preditores e target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identificação de colunas numéricas e categóricas
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Pipeline numérico com imputação pela mediana e padronização
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features)
        ],
        remainder='drop'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, preprocessor