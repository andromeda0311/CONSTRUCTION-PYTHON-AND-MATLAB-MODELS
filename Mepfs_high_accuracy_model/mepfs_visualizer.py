import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def clean_and_prepare_data(filepath):
    """
    Reads and cleans the MEPFS Total Cost CSV data.
    """
    df = pd.read_csv(filepath)

    # Extract Year and Budget from the 'Project' column
    def extract_year_budget(project_str):
        if isinstance(project_str, str):
            # Handle cases like "2017: 12,723,976.07"
            match = re.match(r'(\d{4}).*?([\d,]+\.\d+)', project_str)
            if match:
                year = match.group(1)
                budget = match.group(2).replace(',', '')
                return int(year), float(budget)
        return None, None

    df[['Year_Extracted', 'Budget_Extracted']] = df['Project'].apply(lambda x: pd.Series(extract_year_budget(x)))

    # Use extracted values to fill 'Year' and 'Budget' columns
    df['Year'] = df['Year_Extracted'].fillna(df['Year'])
    df['Budget'] = df['Budget_Extracted'].fillna(df['Budget'])

    # Drop helper columns and the original 'Project' column
    df = df.drop(columns=['Project', 'Year_Extracted', 'Budget_Extracted'])

    # Select columns for visualization (all except Year)
    cost_columns = [col for col in df.columns if col not in ['Year']]
    
    # Convert columns to numeric, coercing errors
    for col in cost_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in cost columns
    df.dropna(subset=cost_columns, inplace=True)
    
    return df, cost_columns

def visualize_data(df, columns, output_dir):
    """
    Generates and saves histograms and boxplots for the given columns.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for col in columns:
        plt.figure(figsize=(12, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)

        plt.tight_layout()
        
        # Sanitize filename
        safe_col_name = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        filename = os.path.join(output_dir, f'{safe_col_name}_visualization.png')
        plt.savefig(filename)
        plt.close()
        print(f"Saved visualization for {col} to {filename}")

if __name__ == "__main__":
    CSV_FILE_PATH = 'Mepfs_high_accuracy_model/MEPFS_Total_Cost.csv'
    VISUALIZATION_DIR = 'Mepfs_high_accuracy_model/visualizations'
    
    data, columns_to_visualize = clean_and_prepare_data(CSV_FILE_PATH)
    
    if not data.empty:
        visualize_data(data, columns_to_visualize, VISUALIZATION_DIR)
        print("\nVisualizations generated successfully.")
    else:
        print("No data available for visualization after cleaning.")
