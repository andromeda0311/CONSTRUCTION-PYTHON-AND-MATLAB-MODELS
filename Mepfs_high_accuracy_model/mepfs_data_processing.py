import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_transformation(original_df, transformed_df, columns, output_dir):
    """
    Generates side-by-side histograms to show the effect of log transformation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for col in columns:
        plt.figure(figsize=(12, 6))

        # Plot original distribution
        plt.subplot(1, 2, 1)
        sns.histplot(original_df[col], kde=True, bins=30)
        plt.title(f'Original Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Plot transformed distribution
        plt.subplot(1, 2, 2)
        sns.histplot(transformed_df[col], kde=True, bins=30)
        plt.title(f'Log-Transformed Distribution of {col}')
        plt.xlabel(f'Log({col})')
        plt.ylabel('Frequency')

        plt.tight_layout()
        
        safe_col_name = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        filename = os.path.join(output_dir, f'{safe_col_name}_normalization_comparison.png')
        plt.savefig(filename)
        plt.close()
        print(f"Saved transformation comparison for {col} to {filename}")

def clean_and_transform_data(input_filepath, output_filepath):
    """
    Reads, cleans, and applies log transformation to the MEPFS Total Cost data.
    """
    df_original = pd.read_csv(input_filepath)

    # --- Data Cleaning (similar to the visualizer script) ---
    def extract_year_budget(project_str):
        if isinstance(project_str, str):
            match = re.match(r'(\d{4}).*?([\d,]+\.\d+)', project_str)
            if match:
                year = match.group(1)
                budget = match.group(2).replace(',', '')
                return int(year), float(budget)
        return None, None

    df_original[['Year_Extracted', 'Budget_Extracted']] = df_original['Project'].apply(lambda x: pd.Series(extract_year_budget(x)))
    df_original['Year'] = df_original['Year_Extracted'].fillna(df_original['Year'])
    df_original['Budget'] = df_original['Budget_Extracted'].fillna(df_original['Budget'])
    df_original = df_original.drop(columns=['Project', 'Year_Extracted', 'Budget_Extracted'])

    cost_columns = [col for col in df_original.columns if col not in ['Year']]
    for col in cost_columns:
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

    df_original.dropna(subset=cost_columns, inplace=True)
    
    # Remove rows where the original budget is 0, as they are not useful for prediction
    original_rows = len(df_original)
    df_original = df_original[df_original['Budget'] > 0]
    rows_removed = original_rows - len(df_original)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with a budget of 0.")

    # Identify feature columns (excluding 'Budget' and 'Year')
    feature_columns = [col for col in df_original.columns if col not in ['Year', 'Budget']]
    
    # Remove rows where Budget > 0 but all feature columns are 0
    original_rows = len(df_original)
    zero_features_mask = (df_original[feature_columns] == 0).all(axis=1)
    df_original = df_original[~zero_features_mask]
    rows_removed = original_rows - len(df_original)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with a budget but all other features are zero.")

    # Create a copy for transformation
    df_transformed = df_original.copy()

    # --- Log Transformation (SKIPPED) ---
    # The log transformation was breaking the additive relationship between features and target,
    # leading to poor model performance. We will proceed with the original, untransformed data.
    print("Skipping log transformation to preserve the natural relationships in the data.")

    # --- Save Preprocessed Data ---
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_transformed.to_csv(output_filepath, index=False)
    print(f"\nPreprocessed data saved to {output_filepath}")
    
    return df_original, df_transformed, cost_columns

if __name__ == "__main__":
    INPUT_CSV_PATH = 'Mepfs_high_accuracy_model/MEPFS_Total_Cost.csv'
    OUTPUT_CSV_PATH = 'Mepfs_high_accuracy_model/mepfs_preprocessed_data_w(log).csv'
    VISUALIZATION_DIR = 'Mepfs_high_accuracy_model/visualizations'
    
    clean_and_transform_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    print("\nData processing complete.")
