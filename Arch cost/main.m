% =============================================================================
% Section 1: Setup Environment
% =============================================================================
clear; clc; close all;
disp('Environment cleared.');

% --- Create a directory for saving visualizations ---
if ~exist('visualizations', 'dir')
   mkdir('visualizations')
   disp("Created 'visualizations' directory.");
end


% =============================================================================
% Section 2: Data Loading and Comprehensive Cleaning
% =============================================================================
disp(newline + "--- Section 2: Data Loading & Cleaning ---");
try
    opts = detectImportOptions('Architectural_Total_Cost.csv');
    opts.VariableNamingRule = 'preserve';
    T = readtable('Architectural_Total_Cost.csv', opts);
    disp("Dataset 'Architectural_Total_Cost.csv' loaded successfully.");
catch ME
    error("Error loading 'Architectural_Total_Cost.csv'. Check file path and format. (%s)", ME.message);
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
    error("Error: No column related to 'budget' found after cleaning headers.");
end
T.Properties.VariableNames{budget_idx} = 'budget';
disp("Detected and standardized budget column: budget");

% --- ROBUST DATA CLEANING AND TYPE CONVERSION ---
if iscell(T.budget)
    T.budget = str2double(strrep(T.budget, ',', ''));
end
T = T(~isnan(T.budget), :);

for i = 1:width(T)
    varName = T.Properties.VariableNames{i};
    if iscell(T.(varName)) && ~strcmp(varName, 'project')
        T.(varName) = str2double(strrep(T.(varName), ',', ''));
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
% Section 3: Enhanced Feature Engineering & Initial Visualizations
% =============================================================================
disp(newline + "--- Section 3: Feature Engineering & Visualizations ---");

% --- VISUALIZATION: Correlation Matrix (Before Feature Engineering) ---
figure('Visible', 'off', 'Position', [100 100 1200 1000]);
corr_matrix_before = corrcoef(T{:, numericVars});
h = heatmap(numericVars, numericVars, corr_matrix_before, 'Colormap', jet);
h.Title = 'Correlation Matrix of Raw Numeric Features (Architectural)';
h.XLabel = 'Features';
h.YLabel = 'Features';
h.CellLabelFormat = '%.2f';
h.FontSize = 8;
saveas(gcf, 'visualizations/correlation_matrix_before_architectural.png');
disp("Saved: correlation_matrix_before_architectural.png");

T.budget_log = log1p(T.budget);
T_clean = T(T.budget > 100000, :);

% --- VISUALIZATION: Budget Distribution (Skewed vs. Log-Transformed) ---
figure('Visible', 'off', 'Position', [100 100 1200 500]);
subplot(1, 2, 1);
histogram(T_clean.budget, 50, 'FaceColor', '#0072BD');
title('Distribution of Budget (Skewed)');
xlabel('Budget');
ylabel('Frequency');
grid on;
subplot(1, 2, 2);
histogram(T_clean.budget_log, 50, 'FaceColor', '#D95319');
title('Distribution of Log-Transformed Budget (Normalized)');
xlabel('Log(Budget + 1)');
ylabel('Frequency');
grid on;
saveas(gcf, 'visualizations/budget_distributions_comparison_architectural.png');
disp("Saved: budget_distributions_comparison_architectural.png");

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

% --- START OF CORRECTION: Dynamically find column names ---
all_vars = T_clean.Properties.VariableNames;

% Find column names related to painting
paint_masonry_col = all_vars{contains(all_vars, 'paintingmasonry')};
paint_wood_col = all_vars{contains(all_vars, 'paintingwood')};
paint_metal_col = all_vars{contains(all_vars, 'paintingmetal')};

% Find column names related to CHB (Concrete Hollow Blocks)
chb_100_col = all_vars{contains(all_vars, 'chb100mm')};
chb_150_col = all_vars{contains(all_vars, 'chb150mm')};

% Find column names related to finishes
plaster_col = all_vars{contains(all_vars, 'plaster')};
tiles_col = all_vars{contains(all_vars, 'glazedtiles')};

% Create new features using the dynamically found column names
T_clean.total_painting_area = T_clean.(paint_masonry_col) + T_clean.(paint_wood_col) + T_clean.(paint_metal_col);
T_clean.total_chb_area = T_clean.(chb_100_col) + T_clean.(chb_150_col);
T_clean.total_finishes_area = T_clean.(plaster_col) + T_clean.(tiles_col);

T_clean.plaster_per_chb = T_clean.(plaster_col) ./ T_clean.total_chb_area;
T_clean.plaster_per_chb(isinf(T_clean.plaster_per_chb) | isnan(T_clean.plaster_per_chb)) = 0;
T_clean.tiles_per_chb = T_clean.(tiles_col) ./ T_clean.total_chb_area;
T_clean.tiles_per_chb(isinf(T_clean.tiles_per_chb) | isnan(T_clean.tiles_per_chb)) = 0;
% --- END OF CORRECTION ---

disp("Feature engineering complete.");


% =============================================================================
% Section 4: Data Preparation for Models
% =============================================================================
disp(newline + "--- Section 4: Preparing Data for Models ---");

% --- MODIFIED: Use the dynamically found column names here as well ---
final_feature_columns = { ...
    'year', 'num_storeys', 'num_classrooms', ...
    plaster_col, tiles_col, ... % Use variables instead of strings
    'total_painting_area', 'total_chb_area', 'total_finishes_area', ...
    'plaster_per_chb', 'tiles_per_chb'
};

disp("Using the following features to prevent data leakage:");
disp(final_feature_columns');

X_final_features = T_clean(:, final_feature_columns);
y = T_clean.budget_log;

% --- VISUALIZATION: Correlation Matrix (After Feature Engineering) ---
figure('Visible', 'off');
corr_matrix_after = corrcoef(table2array(X_final_features));
heatmap(X_final_features.Properties.VariableNames, X_final_features.Properties.VariableNames, corr_matrix_after);
title('Correlation Matrix of Final Engineered Features (Architectural)');
saveas(gcf, 'visualizations/correlation_matrix_after_architectural.png');
disp("Saved: correlation_matrix_after_architectural.png");

cv = cvpartition(height(X_final_features), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);
X_train = X_final_features(idxTrain, :);
y_train = y(idxTrain, :);
X_test = X_final_features(idxTest, :);
y_test = y(idxTest, :);

[X_train_poly, poly_feature_names] = generatePolyFeatures(X_train, 2);
X_test_poly = generatePolyFeatures(X_test, 2);

[X_train_poly_scaled, mu, sigma] = zscore(X_train_poly);
X_test_poly_scaled = (X_test_poly - mu) ./ sigma;
X_test_poly_scaled(isnan(X_test_poly_scaled)) = 0;
disp("Data has been split, transformed, and scaled.");


% =============================================================================
% Section 5: Baseline Gradient Boosting Model
% =============================================================================
disp(newline + "--- Section 5: Baseline Model Performance ---");
baseline_model = fitrensemble(X_train_poly_scaled, y_train, 'Method', 'LSBoost', 'NumLearningCycles', 100);

% --- Evaluate Baseline Model ---
log_preds_base = predict(baseline_model, X_test_poly_scaled);
preds_base = expm1(log_preds_base);
y_test_actual_base = expm1(y_test);
SS_res_base = sum((y_test_actual_base - preds_base).^2);
SS_tot_base = sum((y_test_actual_base - mean(y_test_actual_base)).^2);
r2_base = 1 - (SS_res_base / SS_tot_base);
mae_base = mae(y_test_actual_base - preds_base);
fprintf("Baseline Model R-squared: %.4f\n", r2_base);
fprintf("Baseline Model MAE: %.2f\n", mae_base);


% =============================================================================
% Section 6: Feature Selection
% =============================================================================
disp(newline + "--- Section 6: Feature Selection ---");
importances_base = predictorImportance(baseline_model);

% --- VISUALIZATION: Baseline Feature Importance ---
figure('Visible', 'off');
[sorted_imp, sorted_idx] = sort(importances_base, 'descend');
top_n = 20;
barh(sorted_imp(1:top_n));
yticks(1:top_n);
yticklabels(poly_feature_names(sorted_idx(1:top_n)));
title('Top 20 Feature Importances (Baseline Architectural Model)');
xlabel('Predictor Importance');
set(gca, 'YDir','reverse');
saveas(gcf, 'visualizations/feature_importance_baseline_architectural.png');
disp("Saved: feature_importance_baseline_architectural.png");

threshold = median(importances_base);
selected_mask = importances_base > threshold;
selected_feature_names = poly_feature_names(selected_mask);
X_train_selected = X_train_poly_scaled(:, selected_mask);
X_test_selected = X_test_poly_scaled(:, selected_mask);
fprintf("Feature count after selection: %d\n", size(X_train_selected, 2));


% =============================================================================
% Section 7: Training Optimized Gradient Boosting Model
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
% Section 8: Final Model Evaluation & Visualizations
% =============================================================================
disp(newline + "--- Section 8: Final Model Evaluation & Visualizations ---");
log_predictions = predict(optimized_model, X_test_selected);
final_predictions = expm1(log_predictions);
y_test_actual = expm1(y_test);

SS_res = sum((y_test_actual - final_predictions).^2);
SS_tot = sum((y_test_actual - mean(y_test_actual)).^2);
r2_best = 1 - (SS_res / SS_tot);
mae_best = mae(y_test_actual - final_predictions);
fprintf("Final Optimized Model R-squared: %.4f\n", r2_best);
fprintf("Final Optimized Model MAE: %.2f\n", mae_best);

% --- VISUALIZATION: Actual vs. Predicted ---
figure('Visible', 'off');
scatter(y_test_actual, final_predictions, 'filled');
hold on;
plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], 'r--', 'LineWidth', 2);
title('Actual vs. Predicted Budget (Architectural)');
xlabel('Actual Budget');
ylabel('Predicted Budget');
legend('Predictions', 'Ideal Fit', 'Location', 'northwest');
grid on;
saveas(gcf, 'visualizations/actual_vs_predicted_architectural.png');
disp("Saved: actual_vs_predicted_architectural.png");

% --- VISUALIZATION: Residual Plot ---
residuals = y_test_actual - final_predictions;
figure('Visible', 'off');
scatter(final_predictions, residuals, 'filled');
hold on;
yline(0, 'r--', 'LineWidth', 2);
title('Residual Plot (Architectural)');
xlabel('Predicted Budget');
ylabel('Residuals (Actual - Predicted)');
grid on;
saveas(gcf, 'visualizations/residual_plot_architectural.png');
disp("Saved: residual_plot_architectural.png");

% --- VISUALIZATION: Final Model Feature Importance ---
importances_final = predictorImportance(optimized_model);
figure('Visible', 'off');
[sorted_imp_final, sorted_idx_final] = sort(importances_final, 'descend');
barh(sorted_imp_final(1:top_n));
yticks(1:top_n);
yticklabels(selected_feature_names(sorted_idx_final(1:top_n)));
title('Top 20 Feature Importances (Final Optimized Architectural Model)');
xlabel('Predictor Importance');
set(gca, 'YDir','reverse');
saveas(gcf, 'visualizations/feature_importance_final_architectural.png');
disp("Saved: feature_importance_final_architectural.png");

% --- VISUALIZATION: Performance Comparison ---
figure('Visible', 'off');
metrics_data = [r2_base, r2_best; mae_base, mae_best];
b = bar(metrics_data);
xticklabels({'R-squared', 'MAE'});
legend({'Baseline Model', 'Optimized Model'});
title('Model Performance Comparison (Architectural)');
b(2).FaceColor = '#D95319';
grid on;
saveas(gcf, 'visualizations/performance_comparison_architectural.png');
disp("Saved: performance_comparison_architectural.png");


% =============================================================================
% Section 9: Saving Final Assets
% =============================================================================
disp(newline + "--- Section 9: Saving Final Assets ---");
save('architectural_model_assets.mat', ...
    'optimized_model', ...
    'mu', ...
    'sigma', ...
    'selected_mask', ...
    'final_feature_columns', ...
    'poly_feature_names');
disp("All model assets saved to 'architectural_model_assets.mat'.");
disp(newline + "Process finished.");