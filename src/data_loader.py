import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_clean_data(filepath):
    """
    Loads the dataset, scales 'Amount' and 'Time', and splits into X and y.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data Loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        return None, None

    # Standardize 'Amount' and 'Time' as other features are already PCA transformed
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    X = df.drop('Class', axis=1)
    y = df['Class']

    return X, y


def apply_smote(X_train, y_train):
    """
    Handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
    Ref: Key Work Done
    """
    print("🔄 Applying SMOTE to handle class imbalance...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"✅ Resampling Complete. New shape: {X_res.shape}")
    return X_res, y_res