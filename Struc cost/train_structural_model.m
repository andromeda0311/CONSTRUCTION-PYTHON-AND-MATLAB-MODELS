% =============================================================================
% Section 1: Setup Environment
% =============================================================================
clear; clc; close all;
disp('Environment cleared.');

% =============================================================================
% Section 2: Data Loading and Comprehensive Cleaning
% =============================================================================
disp(newline + "--- Section 2: Data Loading & Cleaning ---");
try
    opts = detectImportOptions('Structural_Total_Cost.csv');
    opts.VariableNamingRule = 'preserve';
    T = readtable('Structural_Total_Cost.csv', opts);
    disp("Dataset 'Structural_Total_Cost.csv' loaded successfully.");
catch ME
    error("Error loading 'Structural_Total_Cost.csv'. Check file path and format. (%s)", ME.message);
end

% --- Clean Column Names ---
validVarNames = matlab.lang.makeValidName(T.Properties.VariableNames);
uniqueVarNames = matlab.lang.makeUniqueStrings(validVarNames);
T.Properties.VariableNames = lower(uniqueVarNames);
disp("Column headers cleaned and duplicates resolved.");
T = T(:, ~startsWith(T.Properties.VariableNames, 'unnamed'));

% --- Identify and Standardize Budget Column ---
budget_idx = find(strcmp(T.Properties.VariableNames, 'budget'), 1);
if isempty(budget_idx)
    budget_idx = find(contains(T.Properties.VariableNames, 'budget', 'IgnoreCase', true), 1);
end
if isempty(budget_idx)
    disp("DEBUG: Cleaned column names are:");
    disp(T.Properties.VariableNames');
    error("Error: No column related to 'budget' found after cleaning headers.");
end
T.Properties.VariableNames{budget_idx} = 'budget';
disp("Detected and standardized budget column: " + string(T.Properties.VariableNames{budget_idx}));

% --- ROBUST DATA CLEANING AND TYPE CONVERSION ---
if iscell(T.budget)
    T.budget = str2double(strrep(T.budget, ',', ''));
end
T = T(~isnan(T.budget), :);

for i = 1:width(T)
    varName = T.Properties.VariableNames{i};
    if iscell(T.(varName)) && ~strcmp(varName, 'project')
        cleaned_cells = strrep(T.(varName), ',', '');
        T.(varName) = str2double(cleaned_cells);
    end
end

% --- ROBUST MISSING VALUE IMPUTATION ---
numericVars = T.Properties.VariableNames(vartype('numeric'));
for i = 1:length(numericVars)
    varName = numericVars{i};
    colMedian = median(T.(varName), 'omitnan');
    T.(varName) = fillmissing(T.(varName), 'constant', colMedian);
end
disp("Data cleaning and type conversion complete.");

% =============================================================================
% Section 3: Enhanced Feature Engineering
% =============================================================================
disp(newline + "--- Section 3: Feature Engineering ---");
T.budget_log = log1p(T.budget);
T_clean = T(T.budget > 100000, :);

tokens_sty = regexp(T_clean.project, '(\d+)\s*STY', 'tokens');
tokens_cl = regexp(T_clean.project, '(\d+)\s*CL', 'tokens');
T_clean.num_storeys = nan(height(T_clean), 1);
T_clean.num_classrooms = nan(height(T_clean), 1);
for i = 1:height(T_clean)
    if ~isempty(tokens_sty{i}); T_clean.num_storeys(i) = str2double(tokens_sty{i}{1}); end
    if ~isempty(tokens_cl{i}); T_clean.num_classrooms(i) = str2double(tokens_cl{i}{1}); end
end

vars_to_fill = {'num_storeys', 'num_classrooms'};
for i = 1:length(vars_to_fill)
    varName = vars_to_fill{i};
    colMedian = median(T_clean.(varName), 'omitnan');
    T_clean.(varName) = fillmissing(T_clean.(varName), 'constant', colMedian);
end

floor_area_cols = T_clean.Properties.VariableNames(contains(T_clean.Properties.VariableNames, 'grossfloorarea'));
T_clean.total_gross_floor_area = sum(T_clean{:, floor_area_cols}, 2);
concrete_cols = T_clean.Properties.VariableNames(contains(T_clean.Properties.VariableNames, 'volumeofstructuralconcrete'));
T_clean.total_concrete_volume = sum(T_clean{:, concrete_cols}, 2);
T_clean.total_reinforcing_steel = T_clean.grade40 + T_clean.grade60;

T_clean.cost_per_sqm = T_clean.budget ./ T_clean.total_gross_floor_area; T_clean.cost_per_sqm(isinf(T_clean.cost_per_sqm)) = 0;
T_clean.cost_per_classroom = T_clean.budget ./ T_clean.num_classrooms; T_clean.cost_per_classroom(isinf(T_clean.cost_per_classroom)) = 0;
T_clean.concrete_per_floor_area = T_clean.total_concrete_volume ./ T_clean.total_gross_floor_area; T_clean.concrete_per_floor_area(isinf(T_clean.concrete_per_floor_area) | isnan(T_clean.concrete_per_floor_area)) = 0;
T_clean.steel_per_concrete = T_clean.total_reinforcing_steel ./ T_clean.total_concrete_volume; T_clean.steel_per_concrete(isinf(T_clean.steel_per_concrete) | isnan(T_clean.steel_per_concrete)) = 0;
disp("Feature engineering complete.");

% =============================================================================
% Section 4: Data Preparation for Models (UPDATED SECTION)
% =============================================================================
disp(newline + "--- Section 4: Preparing Data for Models ---");

% --- Dynamically find the cleaned column name for 'formworks' ---
% This makes the script robust to small changes from the cleaning process.
formworks_col_idx = find(contains(T_clean.Properties.VariableNames, 'formworks'));
if isempty(formworks_col_idx)
    error('FATAL: Could not find any column related to "formworks" in the cleaned data.');
elseif numel(formworks_col_idx) > 1
    warning('Multiple columns containing "formworks" were found. Using the first one: %s', T_clean.Properties.VariableNames{formworks_col_idx(1)});
end
formworks_col_name = T_clean.Properties.VariableNames{formworks_col_idx(1)};
fprintf("Dynamically identified formworks column as: '%s'\n", formworks_col_name);

% Define the final feature list, using the dynamically found formworks column name
final_feature_columns = { ...
    'year', 'num_storeys', 'num_classrooms', 'total_gross_floor_area', ...
    'total_concrete_volume', 'total_reinforcing_steel', formworks_col_name, ... % <-- DYNAMIC NAME USED HERE
    'cost_per_sqm', 'cost_per_classroom', 'concrete_per_floor_area', 'steel_per_concrete'
};
X = T_clean(:, final_feature_columns);
y = T_clean.budget_log;

cv = cvpartition(height(X), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);
X_train = X(idxTrain, :);
y_train = y(idxTrain, :);
X_test = X(idxTest, :);
y_test = y(idxTest, :);

% This function needs to be defined or available on your MATLAB path
% Assuming generatePolyFeatures exists as a separate function
[X_train_poly, poly_feature_names] = generatePolyFeatures(X_train, 2);
X_test_poly = generatePolyFeatures(X_test, 2);

[X_train_poly_scaled, mu, sigma] = zscore(X_train_poly);
X_test_poly_scaled = (X_test_poly - mu) ./ sigma;
X_test_poly_scaled(isnan(X_test_poly_scaled)) = 0;
disp("Data has been split, transformed, and scaled.");

% =============================================================================
% Section 5: Baseline Gradient Boosting Model (No changes)
% =============================================================================
disp(newline + "--- Section 5: Baseline Model Performance ---");
baseline_model = fitrensemble(X_train_poly_scaled, y_train, 'Method', 'LSBoost', 'NumLearningCycles', 100);

% =============================================================================
% Section 6: Feature Selection (No changes)
% =============================================================================
disp(newline + "--- Section 6: Feature Selection ---");
importances = predictorImportance(baseline_model);
threshold = median(importances);
selected_mask = importances > threshold;
selected_feature_names = poly_feature_names(selected_mask);
X_train_selected = X_train_poly_scaled(:, selected_mask);
X_test_selected = X_test_poly_scaled(:, selected_mask);
fprintf("Feature count after selection: %d\n", size(X_train_selected, 2));

% =============================================================================
% Section 7: Training Optimized Gradient Boosting Model (No changes)
% =============================================================================
disp(newline + "--- Section 7: Training Optimized Model ---");
optimizable_vars = [
    optimizableVariable('NumLearningCycles', [500, 1000], 'Type', 'integer');
    optimizableVariable('MaxNumSplits', [8, 32], 'Type', 'integer');
    optimizableVariable('LearnRate', [0.01, 0.1], 'Type', 'real', 'Transform', 'log');
];
optimized_model = fitrensemble(X_train_selected, y_train, ...
    'Method', 'LSBoost', ...
    'OptimizeHyperparameters', optimizable_vars, ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'ShowPlots', false, 'Verbose', 1));

% =============================================================================
% Section 8: Final Model Evaluation (No changes)
% =============================================================================
disp(newline + "--- Section 8: Final Model Evaluation ---");
log_predictions = predict(optimized_model, X_test_selected);
final_predictions = expm1(log_predictions);
y_test_actual = expm1(y_test);

SS_res = sum((y_test_actual - final_predictions).^2);
SS_tot = sum((y_test_actual - mean(y_test_actual)).^2);
r2_best = 1 - (SS_res / SS_tot);
mae_best = mae(y_test_actual - final_predictions);
fprintf("Final Optimized Model R-squared: %.4f\n", r2_best);
fprintf("Final Optimized Model MAE: %.2f\n", mae_best);

% =============================================================================
% Section 9: Saving Final Assets (No changes)
% =============================================================================
disp(newline + "--- Section 9: Saving Final Assets ---");
save('structural_model_assets.mat', ...
    'optimized_model', ...
    'mu', ...
    'sigma', ...
    'selected_mask', ...
    'final_feature_columns', ...
    'poly_feature_names');
disp("All model assets saved to 'structural_model_assets.mat'.");
disp(newline + "Process finished.");