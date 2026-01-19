from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


def train_logistic_regression(X_train, y_train):
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest (using all CPU cores)...")
    # n_jobs=-1 uses all available processors
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates model using Confusion Matrix, ROC-AUC, and Precision-Recall.
    Ref: Key Work Done
    """
    y_pred = model.predict(X_test)

    print(f"\n--- Evaluation for {model_name} ---")
    # This prints Precision, Recall, F1-Score
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, y_pred)
    print(f"ROC-AUC Score: {roc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    return cm


def tune_hyperparameters(X_train, y_train):
    """
    Optimizes Random Forest using RandomizedSearchCV.
    Ref: Key Work Done - Optimized model with hyperparameter tuning
    """
    print("\n⚡ Optimizing Hyperparameters for Random Forest...")

    # Define grid of parameters to test
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)

    # Randomized Search is faster than Grid Search
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=5,  # Number of parameter settings that are sampled
        cv=3,  # 3-fold cross-validation
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print(f"✅ Best Parameters Found: {random_search.best_params_}")
    return random_search.best_estimator_