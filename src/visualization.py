import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_class_distribution(y):
    """Visualizes the imbalance between Fraud vs Normal transactions."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.show()


def plot_correlation_matrix(df):
    """
    Plots heatmap to identify important features.
    Ref: Key Work Done - Applied correlation analysis
    """
    plt.figure(figsize=(14, 10))
    # Correlation matrix of the entire dataset
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot=False)
    plt.title('Feature Correlation Matrix')
    plt.show()


def plot_confusion_matrix(cm, model_name):
    """Visualizes the confusion matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()


def plot_dim_reduction(X, y):
    """
    Visualizes fraud vs non-fraud patterns using PCA and t-SNE.
    Ref: Key Work Done - Visualized patterns using PCA/TSNE
    NOTE: We take a sample because t-SNE is very slow on large datasets.
    """
    print("⏳ Generating PCA & t-SNE plots (taking a sample of 2000 points)...")

    # Take a random sample to speed up visualization
    sample_size = 2000
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
        y_sample = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
    else:
        X_sample, y_sample = X, y

    # 1. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)

    # 2. t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # PCA Plot
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_sample, palette=['blue', 'red'], alpha=0.6, ax=ax1)
    ax1.set_title("PCA: Fraud vs Non-Fraud Patterns")

    # t-SNE Plot
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_sample, palette=['blue', 'red'], alpha=0.6, ax=ax2)
    ax2.set_title("t-SNE: Fraud vs Non-Fraud Patterns")

    plt.show()