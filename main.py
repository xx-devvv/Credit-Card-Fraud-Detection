from src.data_loader import load_and_clean_data, apply_smote
from src.visualization import plot_class_distribution, plot_confusion_matrix, plot_correlation_matrix, \
    plot_dim_reduction
from src.model import train_logistic_regression, train_random_forest, train_xgboost, evaluate_model, \
    tune_hyperparameters
from sklearn.model_selection import train_test_split

# CONFIG
DATA_PATH = "data/creditcard.csv"


def main():
    # --- STEP 1: LOAD & EDA ---
    print("--- STEP 1: Loading & EDA ---")
    X, y = load_and_clean_data(DATA_PATH)
    if X is None: return

    # 1. Visualize Imbalance [cite: 22]
    plot_class_distribution(y)

    # 2. Correlation Analysis
    # (We temporarily combine X and y for the heatmap)
    df_temp = X.copy()
    df_temp['Class'] = y
    plot_correlation_matrix(df_temp)

    # 3. PCA & t-SNE Visualization
    plot_dim_reduction(X, y)

    # --- STEP 2: PREPROCESSING ---
    print("\n--- STEP 2: Preprocessing ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle Class Imbalance using SMOTE [cite: 21]
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # --- STEP 3: TRAINING & EVALUATION ---
    print("\n--- STEP 3: Training & Evaluation ---")

    # Train Logistic Regression [cite: 24]
    lr_model = train_logistic_regression(X_train_smote, y_train_smote)
    cm_lr = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    plot_confusion_matrix(cm_lr, "Logistic Regression")

    # Train Random Forest [cite: 24]
    rf_model = train_random_forest(X_train_smote, y_train_smote)
    cm_rf = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    plot_confusion_matrix(cm_rf, "Random Forest")

    # Train XGBoost [cite: 24]
    xgb_model = train_xgboost(X_train_smote, y_train_smote)
    cm_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    plot_confusion_matrix(cm_xgb, "XGBoost")

    # --- STEP 4: OPTIMIZATION ---
    print("\n--- STEP 4: Hyperparameter Tuning ---")
    # Optimize Random Forest as an example
    best_rf_model = tune_hyperparameters(X_train_smote, y_train_smote)
    cm_best = evaluate_model(best_rf_model, X_test, y_test, "Optimized Random Forest")
    plot_confusion_matrix(cm_best, "Optimized Random Forest")


if __name__ == "__main__":
    main()