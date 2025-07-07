# utils.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score, roc_auc_score, roc_curve
)


def load_data():
    return pd.read_csv("heart.csv")

def preprocess(df):
    df = df.drop_duplicates()

    # Outlier handling (es. Age)
    Q1 = df["Age"].quantile(0.25)
    Q3 = df["Age"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df["Age"] < (Q1 - 1.5 * IQR)) | (df["Age"] > (Q3 + 1.5 * IQR)))]

    # Replace Cholesterol = 0 con with the mean
    mean_chol = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
    df['Cholesterol'] = df['Cholesterol'].replace(0, mean_chol)

    y = df['HeartDisease']
    X = df.drop('HeartDisease', axis=1)
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def train_model(X_train, y_train, model, param_grid, cv):
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='recall', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print(f"Best CV recall: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)
    print(f"Average Precision (area under P-R curve): {avg_precision:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP = {avg_precision:.2f}", color='blue')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({model.__class__.__name__})")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    auc = roc_auc_score(y_test, y_scores)
    print(f"AUC: {auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    for i in range(0, len(thresholds), 10):
        plt.annotate(f'{thresholds[i]:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model.__class__.__name__})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_feature_importance(model, X):
    importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.show()
