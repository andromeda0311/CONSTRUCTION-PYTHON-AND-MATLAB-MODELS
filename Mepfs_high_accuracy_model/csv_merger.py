import pandas as pd
import re
import os

def standardize_project_name(name):
    """
    Standardizes project names to the 'X STY Y CLS Location' format.
    Handles formats like '2x4', '2 STY 4 CL', etc., and preserves the location.
    """
    if not isinstance(name, str):
        return name
    
    original_name = name
    name_upper = name.upper()
    
    # Pattern for STY and CL/CLS
    sty_cl_match = re.search(r'(\d+)\s*STY\s*(\d+)\s*CL(S)?', name_upper)
    if sty_cl_match:
        floors = sty_cl_match.group(1)
        rooms = sty_cl_match.group(2)
        # Get the text that follows the matched pattern
        rest_of_name = original_name[sty_cl_match.end():].strip()
        return f"{floors} STY {rooms} CLS {rest_of_name}"

    # Pattern for 'x' separator
    x_match = re.search(r'(\d+)\s*X\s*(\d+)', name_upper)
    if x_match:
        floors = x_match.group(1)
        rooms = x_match.group(2)
        rest_of_name = original_name[x_match.end():].strip()
        return f"{floors} STY {rooms} CLS {rest_of_name}"

    return original_name # Return original name if no pattern is matched

def extract_year_and_budget(entry):
    """
    Extracts the year and budget from a string.
    Handles 'YYYY: budget' format, just budget numbers, and malformed numbers.
    """
    if not isinstance(entry, str):
        return None, None

    # Pre-process the entry to handle formats like "2020.11.842,000.00"
    cleaned_entry = re.sub(r'(\d{4})\.', r'\1: ', entry, count=1)
    
    # Regex for 'YYYY: budget' format
    match = re.match(r'(\d{4}|\d{2}//):\s*(.*)', cleaned_entry.strip())
    
    if match:
        year_str = match.group(1).replace('//', '00')
        budget_str = match.group(2).replace(',', '')

        # Handle multiple periods by keeping only the last one as a decimal
        if budget_str.count('.') > 1:
            parts = budget_str.split('.')
            budget_str = "".join(parts[:-1]) + "." + parts[-1]
            
        try:
            budget = float(budget_str)
            return year_str, budget
        except (ValueError, TypeError):
            return year_str, None

    # If no year is found, try to convert the whole (cleaned) string to a budget number
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
    """
    Reads quantity and unit cost CSVs, processes them, and saves the merged result.
    """
    try:
        quantity_df = pd.read_csv(quantity_file)
        unit_cost_df = pd.read_csv(unit_cost_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the input files are in the script's directory.")
        return

    # --- Data Cleaning and Preparation ---
    # Rename columns for clarity
    quantity_df.rename(columns={quantity_df.columns[0]: 'Project', quantity_df.columns[1]: 'Year_Budget'}, inplace=True)
    unit_cost_df.rename(columns={unit_cost_df.columns[0]: 'Project', unit_cost_df.columns[1]: 'Year_Budget'}, inplace=True)

    # Store original project names to ensure a perfect merge before standardization
    quantity_df['Original_Project'] = quantity_df['Project']
    
    # Extract Year and Budget
    year_budget_info = quantity_df['Year_Budget'].apply(extract_year_and_budget).apply(pd.Series)
    quantity_df['Year'] = year_budget_info[0]
    quantity_df['Budget'] = year_budget_info[1]

    # Use a stable key for merging
    quantity_df.set_index('Original_Project', inplace=True)
    unit_cost_df.set_index('Project', inplace=True)

    # --- Calculation ---
    # Isolate only the numeric columns for multiplication
    start_col_index_qty = quantity_df.columns.get_loc('MEPFS aspect') + 1
    numeric_quantity = quantity_df.iloc[:, start_col_index_qty:-2].copy()

    start_col_index_unit = unit_cost_df.columns.get_loc('MEPFS aspect') + 1
    numeric_unit_cost = unit_cost_df.iloc[:, start_col_index_unit:].copy()
    
    # Align columns to ensure correct element-wise multiplication
    numeric_unit_cost = numeric_unit_cost.reindex(columns=numeric_quantity.columns)

    # Convert all relevant columns to numeric, coercing errors to NaN, then filling with 0
    for col in numeric_quantity.columns:
        numeric_quantity[col] = pd.to_numeric(numeric_quantity[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    for col in numeric_unit_cost.columns:
        numeric_unit_cost[col] = pd.to_numeric(numeric_unit_cost[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # Calculate the total cost for each structural aspect
    total_cost_df = numeric_quantity.multiply(numeric_unit_cost)

    # --- Final Assembly ---
    # Combine the identifying columns with the calculated total costs
    final_df = quantity_df[['Project', 'Year', 'Budget']].copy()
    
    # Apply name standardization to the final 'Project' column
    final_df['Project'] = final_df['Project'].apply(standardize_project_name)
    
    final_df = final_df.join(total_cost_df)
    final_df.reset_index(drop=True, inplace=True)

    # Reorder columns to have Project, Year, and Budget at the beginning
    cols = ['Project', 'Year', 'Budget'] + [col for col in final_df if col not in ['Project', 'Year', 'Budget']]
    final_df = final_df[cols]

    # Save the final dataframe to a new CSV file
    final_df.to_csv(output_file, index=False)
    print(f"Successfully created the merged file: {output_file}")


# --- Execution ---
# Define the names of your input and output files.
quantity_filename = 'Mepfs_high_accuracy_model/MEPFS Quantity Cost.csv'
unit_cost_filename = 'Mepfs_high_accuracy_model/MEPFS Unit Cost.csv'
output_filename = 'Mepfs_high_accuracy_model/MEPFS_Total_Cost.csv'

# Run the main processing function
process_cost_files(quantity_filename, unit_cost_filename, output_filename)
