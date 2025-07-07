# main.py

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, preprocess, train_model, evaluate_model, plot_feature_importance

def main():
    df = load_data()
    X, y = preprocess(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling for lr
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear']
    }
    best_lr = train_model(X_train_scaled, y_train, lr_model, lr_param_grid, cv)

    print("\nEvaluation Logistic Regression:")
    evaluate_model(best_lr, X_test_scaled, y_test)

    # Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    best_rf = train_model(X_train, y_train, rf_model, rf_param_grid, cv)

    print("\nEvaluation Random Forest:")
    evaluate_model(best_rf, X_test, y_test)

    # Feature importance
    plot_feature_importance(best_rf, X)

if __name__ == "__main__":
    main()