"""
Model Training & Evaluation Script
Author: Aryaman Srimali Jain (Zenecks Infotech ML Internship)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Placement_Data.csv"
MODEL_OUT = BASE_DIR / "models" / "placement_predictor_pipeline.joblib"


def train_models():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[1/4] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} features")

    # Features and Target
    X_df = df.drop(columns=['sl_no', 'salary', 'status']).copy()
    y = (df['status'] == 'Placed').astype(int).values

    # Encode categorical features
    cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col])
        label_encoders[col] = le

    feature_names = list(X_df.columns)
    X = X_df.values

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[2/4] Benchmarking models with 5-Fold Stratified Cross-Validation...")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_f1 = -1

    for name, model in models.items():
        X_tr = X_train_scaled if name in ['Logistic Regression', 'Support Vector Machine'] else X_train
        X_te = X_test_scaled if name in ['Logistic Regression', 'Support Vector Machine'] else X_test

        cv_scores = cross_validate(model, X_tr, y_train, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"  • {name:24s} | Acc: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f} | CV Acc: {cv_scores['test_accuracy'].mean():.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = {
                'name': name,
                'model': model,
                'scaler': scaler,
                'label_encoders': label_encoders,
                'feature_names': feature_names,
                'metrics': {'test_accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'roc_auc': auc}
            }

    print(f"\n[3/4] Best Performing Model: {best_model['name']} (F1: {best_f1:.3f})")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_OUT)
    print(f"[4/4] Saved best model pipeline to {MODEL_OUT}")


if __name__ == "__main__":
    train_models()
