#!/usr/bin/env python
# coding: utf-8

# # MEPFS Budget Prediction Workflow (End-to-End)
# 
# This notebook consolidates the entire workflow for the MEPFS budget prediction model, including data merging, initial visualization, data processing, model training, and evaluation.

# ## 1. Setup and Imports

# In[ ]:


import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))


# ## 2. Data Merging (from `csv_merger.py`)
# 
# First, we merge the quantity and unit cost CSV files to create a total cost file.

# In[ ]:


def standardize_project_name(name):
    if not isinstance(name, str):
        return name
    original_name = name
    name_upper = name.upper()
    sty_cl_match = re.search(r'(\d+)\s*STY\s*(\d+)\s*CL(S)?', name_upper)
    if sty_cl_match:
        floors = sty_cl_match.group(1)
        rooms = sty_cl_match.group(2)
        rest_of_name = original_name[sty_cl_match.end():].strip()
        return f"{floors} STY {rooms} CLS {rest_of_name}"
    x_match = re.search(r'(\d+)\s*X\s*(\d+)', name_upper)
    if x_match:
        floors = x_match.group(1)
        rooms = x_match.group(2)
        rest_of_name = original_name[x_match.end():].strip()
        return f"{floors} STY {rooms} CLS {rest_of_name}"
    return original_name

def extract_year_and_budget(entry):
    if not isinstance(entry, str):
        return None, None
    cleaned_entry = re.sub(r'(\d{4})\.', r'\1: ', entry, count=1)
    match = re.match(r'(\d{4}|\d{2}//):\s*(.*)', cleaned_entry.strip())
    if match:
        year_str = match.group(1).replace('//', '00')
        budget_str = match.group(2).replace(',', '')
        if budget_str.count('.') > 1:
            parts = budget_str.split('.')
            budget_str = "".join(parts[:-1]) + "." + parts[-1]
        try:
            budget = float(budget_str)
            return year_str, budget
        except (ValueError, TypeError):
            return year_str, None
    try:
        budget_str = cleaned_entry.replace(',', '')
        if budget_str.count('.') > 1:
            parts = budget_str.split('.')
            budget_str = "".join(parts[:-1]) + "." + parts[-1]
        budget = float(budget_str)
        return None, budget
    except (ValueError, TypeError):
        return None, None

def process_cost_files(quantity_file, unit_cost_file, output_file):
    quantity_df = pd.read_csv(quantity_file)
    unit_cost_df = pd.read_csv(unit_cost_file)
    quantity_df.rename(columns={quantity_df.columns[0]: 'Project', quantity_df.columns[1]: 'Year_Budget'}, inplace=True)
    unit_cost_df.rename(columns={unit_cost_df.columns[0]: 'Project', unit_cost_df.columns[1]: 'Year_Budget'}, inplace=True)
    quantity_df['Original_Project'] = quantity_df['Project']
    year_budget_info = quantity_df['Year_Budget'].apply(extract_year_and_budget).apply(pd.Series)
    quantity_df['Year'] = year_budget_info[0]
    quantity_df['Budget'] = year_budget_info[1]
    quantity_df.set_index('Original_Project', inplace=True)
    unit_cost_df.set_index('Project', inplace=True)
    start_col_index_qty = quantity_df.columns.get_loc('MEPFS aspect') + 1
    numeric_quantity = quantity_df.iloc[:, start_col_index_qty:-2].copy()
    start_col_index_unit = unit_cost_df.columns.get_loc('MEPFS aspect') + 1
    numeric_unit_cost = unit_cost_df.iloc[:, start_col_index_unit:].copy()
    numeric_unit_cost = numeric_unit_cost.reindex(columns=numeric_quantity.columns)
    for col in numeric_quantity.columns:
        numeric_quantity[col] = pd.to_numeric(numeric_quantity[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    for col in numeric_unit_cost.columns:
        numeric_unit_cost[col] = pd.to_numeric(numeric_unit_cost[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    total_cost_df = numeric_quantity.multiply(numeric_unit_cost)
    final_df = quantity_df[['Project', 'Year', 'Budget']].copy()
    final_df['Project'] = final_df['Project'].apply(standardize_project_name)
    final_df = final_df.join(total_cost_df)
    final_df.reset_index(drop=True, inplace=True)
    cols = ['Project', 'Year', 'Budget'] + [col for col in final_df if col not in ['Project', 'Year', 'Budget']]
    final_df = final_df[cols]
    # Clean up unnecessary columns
    final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]
    final_df.dropna(axis=1, how='all', inplace=True)
    print(f'Cleaned columns. Remaining columns: {final_df.columns.tolist()}')
    final_df.to_csv(output_file, index=False)
    print(f"Successfully created the merged file: {output_file}")
    return final_df

quantity_filename = os.path.join(script_dir, 'MEPFS Quantity Cost.csv')
unit_cost_filename = os.path.join(script_dir, 'MEPFS Unit Cost.csv')
total_cost_filename = os.path.join(script_dir, 'MEPFS_Total_Cost.csv')
total_cost_df = process_cost_files(quantity_filename, unit_cost_filename, total_cost_filename)


# ## 3. Initial Data Visualization (from `mepfs_visualizer.py`)
# 
# Now, let's visualize the distributions of the merged data to check for outliers and skewness.

# In[ ]:


def visualize_distributions(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cost_columns = [col for col in df.columns if col not in ['Year', 'Project']]
    for col in cost_columns:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Histogram of {col}')
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col].dropna())
        plt.title(f'Boxplot of {col}')
        safe_col_name = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        filename = os.path.join(output_dir, f'{safe_col_name}_initial_visualization.png')
        plt.savefig(filename)
        plt.show()
        print(f"Saved initial visualization for {col} to {filename}")

VISUALIZATION_DIR = os.path.join(script_dir, 'visualizations')
visualize_distributions(total_cost_df, VISUALIZATION_DIR)


# ## 4. Data Preprocessing (from `mepfs_data_processing.py`)
# 
# Next, we clean the merged data and filter out irrelevant rows for model training.

# In[ ]:


def clean_and_transform_data(df, output_filepath):
    df_processed = df.copy()
    cost_columns = [col for col in df_processed.columns if col not in ['Year', 'Project']]
    for col in cost_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed.dropna(subset=cost_columns, inplace=True)
    df_processed = df_processed[df_processed['Budget'] > 0]
    feature_columns = [col for col in df_processed.columns if col not in ['Year', 'Budget', 'Project']]
    zero_features_mask = (df_processed[feature_columns] == 0).all(axis=1)
    df_processed = df_processed[~zero_features_mask]
    df_processed = df_processed.drop(columns=['Project'])
    df_processed.to_csv(output_filepath, index=False)
    print(f"\nPreprocessed data saved to {output_filepath}")
    return df_processed

PREPROCESSED_DATA_PATH = os.path.join(script_dir, 'mepfs_preprocessed_data.csv')
preprocessed_df = clean_and_transform_data(total_cost_df, PREPROCESSED_DATA_PATH)


# ## 5. Model Training and Advanced Visualization

# In[ ]:


def plot_correlation_matrix(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    filename = os.path.join(output_dir, 'feature_correlation_matrix.png')
    plt.savefig(filename)
    plt.show()
    print(f"Correlation matrix saved to {filename}")

def plot_actual_vs_predicted(y_true, y_pred, output_dir, data_set_name='Test'):
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
    plt.show()
    print(f"Actual vs. Predicted plot for {data_set_name} set saved to {filename}")

def plot_loss_history(evals_result, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(10, 6))
    plt.plot(evals_result['training']['l2'], label='Training Loss')
    plt.plot(evals_result['valid_1']['l2'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    filename = os.path.join(output_dir, 'epoch_history.png')
    plt.savefig(filename)
    plt.show()
    print(f"Epoch history plot saved to {filename}")

def train_budget_prediction_model(df, model_output_dir, visualization_dir):
    plot_correlation_matrix(df.drop('Year', axis=1, errors='ignore'), visualization_dir)
    X = df.drop('Budget', axis=1)
    y = df['Budget']
    if 'Year' in X.columns:
        X = X.drop('Year', axis=1)
    print(f"Features: {', '.join(X.columns)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # Build and train the LightGBM model
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    evals_result = {}
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=180,
                      valid_sets=[lgb_train, lgb_eval],
                      callbacks=[lgb.log_evaluation(period=1), lgb.record_evaluation(evals_result)])
    
    print("Model training complete.")

    # Plot training history
    plot_loss_history(evals_result, visualization_dir)

    # Evaluate the model
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Training Accuracy (R²): {train_r2:.2%}")

    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Testing Accuracy (R²): {test_r2:.2%}")

    plot_actual_vs_predicted(y_train, y_train_pred, visualization_dir, data_set_name='Training')
    plot_actual_vs_predicted(y_test, y_test_pred, visualization_dir, data_set_name='Testing')

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    model_filepath = os.path.join(model_output_dir, 'mepfs_budget_model_lgbm.joblib')
    joblib.dump(model, model_filepath)
    print(f"\nTrained model saved to {model_filepath}")

MODEL_OUTPUT_DIR = os.path.join(script_dir, 'models')
train_budget_prediction_model(preprocessed_df, MODEL_OUTPUT_DIR, VISUALIZATION_DIR)
