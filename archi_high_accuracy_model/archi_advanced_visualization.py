import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

def plot_correlation_matrix(df, output_dir):
    """
    Generates and saves a correlation matrix heatmap for the given DataFrame.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Architectural Feature Correlation Matrix')
    
    filename = os.path.join(output_dir, 'architectural_correlation_matrix.png')
    plt.savefig(filename)
    plt.close()
    print(f"Correlation matrix saved to {filename}")

def plot_actual_vs_predicted(y_true, y_pred, output_dir, data_set_name='Test'):
    """
    Generates a scatter plot of actual vs. predicted values.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs. Predicted Values ({data_set_name} Set)')
    plt.grid(True)
    
    filename = os.path.join(output_dir, f'architectural_{data_set_name.lower()}_actual_vs_predicted.png')
    plt.savefig(filename)
    plt.close()
    print(f"Actual vs. Predicted plot for {data_set_name} set saved to {filename}")

if __name__ == "__main__":
    # --- Load Data and Model ---
    PREPROCESSED_DATA_PATH = 'archi_high_accuracy_model/architectural_cleaned.csv'
    MODEL_PATH = 'archi_high_accuracy_model/models/archi_budget_model.joblib'
    VISUALIZATION_DIR = 'archi_high_accuracy_model/visualizations'
    
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    model = joblib.load(MODEL_PATH)
    
    # --- Generate Correlation Matrix ---
    plot_correlation_matrix(df.drop('Year', axis=1, errors='ignore'), VISUALIZATION_DIR)
    
    # --- Prepare Data for Prediction ---
    X = df.drop('Budget', axis=1)
    y = df['Budget']
    
    if 'Year' in X.columns:
        X = X.drop('Year', axis=1)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Generate Predictions ---
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # --- Generate Plots ---
    plot_actual_vs_predicted(y_train, y_train_pred, VISUALIZATION_DIR, data_set_name='Training')
    plot_actual_vs_predicted(y_test, y_test_pred, VISUALIZATION_DIR, data_set_name='Testing')
