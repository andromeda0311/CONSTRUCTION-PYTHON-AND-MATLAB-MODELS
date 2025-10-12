% MATLAB Script for Data Visualization and Model Retraining

% --- 1. Data Loading and Preparation ---
disp('Step 1: Loading and preparing data...');
opts = detectImportOptions('architectural_cleaned.csv');
opts.VariableNamingRule = 'preserve'; % Preserve original column headers
data = readtable('architectural_cleaned.csv', opts);

% Remove the 'Year' column
data.Year = [];
disp('     - ''Year'' column removed.');

% Separate features and target
features = data(:, 1:end-1);
target = data(:, end);

% Convert table to array for modeling
X = features{:,:};
y = target{:,:};

% Normalize the data
[X_norm, PS] = mapstd(X');
[y_norm, TS] = mapstd(y');
disp('     - Data normalization complete.');

% --- 2. Create Visualizations Directory ---
if ~exist('visualizations', 'dir')
   mkdir('visualizations');
   disp('Step 2: Created ''visualizations'' directory.');
else
   disp('Step 2: ''visualizations'' directory already exists.');
end

% --- 3. Data Visualization ---
disp('Step 3: Generating and saving visualizations...');

% Correlation Matrix
figure('Visible', 'off'); % Create figure in the background
corr_matrix = corrcoef(X);
clean_labels = strrep(features.Properties.VariableNames, '_', ' ');
heatmap(clean_labels, clean_labels, corr_matrix);
title('Correlation Matrix of Features (Without Year)');
saveas(gcf, 'visualizations/architectural_correlation_matrix.png');
disp('     - Correlation matrix saved.');

% --- 4. Model Training (Neural Network) ---
disp('Step 4: Training neural network model...');

% Define the network architecture with two hidden layers
net = feedforwardnet([10 5]);

% Set up data division for training, validation, and testing (70/15/15 split)
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Show training window with epoch progress
net.trainParam.showWindow = true;

% Train the network
[net,tr] = train(net, X_norm, y_norm);
disp('     - Model training complete.');

% --- 5. Model Evaluation and Visualization ---
disp('Step 5: Evaluating model and generating results...');
% Make predictions on the entire dataset (normalized)
y_pred_norm = net(X_norm);

% Reverse normalization to get predictions in original scale
y_pred = mapstd('reverse', y_pred_norm, TS);

% Separate original and predicted values for train, validation, and test sets
y_train = y(tr.trainInd)';
y_val = y(tr.valInd)';
y_test = y(tr.testInd)';
y_pred_train = y_pred(tr.trainInd);
y_pred_val = y_pred(tr.valInd);
y_pred_test = y_pred(tr.testInd);

% Calculate performance metrics
R2_train = 1 - sum((y_train - y_pred_train).^2) / sum((y_train - mean(y_train)).^2);
RMSE_train = sqrt(mean((y_train - y_pred_train).^2));
R2_test = 1 - sum((y_test - y_pred_test).^2) / sum((y_test - mean(y_test)).^2);
RMSE_test = sqrt(mean((y_test - y_pred_test).^2));

% Display metrics in command window
disp('------------------------------------------');
disp('           Model Performance');
disp('------------------------------------------');
fprintf('Training R-squared: %.4f\n', R2_train);
fprintf('Training RMSE: %.2f\n', RMSE_train);
fprintf('Testing R-squared: %.4f\n', R2_test);
fprintf('Testing RMSE: %.2f\n', RMSE_test);
disp('------------------------------------------');

% Epoch performance plot
figure('Visible', 'off');
plotperform(tr);
saveas(gcf, 'visualizations/epoch_performance.png');
disp('     - Epoch performance plot saved.');

% Actual vs. Predicted (Training Set)
figure('Visible', 'off');
scatter(y_train, y_pred_train, 'filled');
hold on;
plot(y_train, y_train, 'r-', 'LineWidth', 2);
hold off;
xlabel('Actual Values');
ylabel('Predicted Values');
title('Training: Actual vs. Predicted Values');
grid on;
saveas(gcf, 'visualizations/architectural_training_actual_vs_predicted.png');
disp('     - Training performance plot saved.');

% Actual vs. Predicted (Testing Set)
figure('Visible', 'off');
scatter(y_test, y_pred_test, 'filled');
hold on;
plot(y_test, y_test, 'r-', 'LineWidth', 2);
hold off;
xlabel('Actual Values');
ylabel('Predicted Values');
title('Testing: Actual vs. Predicted Values');
grid on;
saveas(gcf, 'visualizations/architectural_testing_actual_vs_predicted.png');
disp('     - Testing performance plot saved.');

% Residual Plot
residuals = y_test - y_pred_test;
figure('Visible', 'off');
scatter(y_pred_test, residuals, 'filled');
hold on;
yline(0, 'r-', 'LineWidth', 2);
hold off;
xlabel('Predicted Values');
ylabel('Residuals');
title('Residual Plot');
grid on;
saveas(gcf, 'visualizations/architectural_residual_plot.png');
disp('     - Residual plot saved.');

disp('--- All processes complete. ---');
