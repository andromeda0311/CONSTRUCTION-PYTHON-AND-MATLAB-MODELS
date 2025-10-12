import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_correlation_matrix(df, output_dir):
    """
    Generates and saves a correlation matrix heatmap for the given DataFrame.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    
    filename = os.path.join(output_dir, 'feature_correlation_matrix.png')
    plt.savefig(filename)
    plt.close()
    print(f"Correlation matrix saved to {filename}")

def plot_epoch_history(history, output_dir, train_or_test='training'):
    """
    Plots the model's performance over epochs and saves the image.
    This is a placeholder and will need to be adapted based on the specific model's training history object.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    # Example for a Keras-like history object
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title(f'Model Loss During {train_or_test.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    filename = os.path.join(output_dir, f'{train_or_test}_epoch_history.png')
    plt.savefig(filename)
    plt.close()
    print(f"Epoch history for {train_or_test} saved to {filename}")

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
    
    filename = os.path.join(output_dir, f'{data_set_name.lower()}_actual_vs_predicted.png')
    plt.savefig(filename)
    plt.close()
    print(f"Actual vs. Predicted plot for {data_set_name} set saved to {filename}")
