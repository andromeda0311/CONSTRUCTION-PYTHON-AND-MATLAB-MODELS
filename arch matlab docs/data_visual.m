% MATLAB Script for Data Visualization

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
disp('     - Data preparation complete.');

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

% Budget Distribution
figure('Visible', 'off');
histogram(y, 'Normalization', 'pdf');
title('Distribution of Budget');
xlabel('Budget');
ylabel('Density');
grid on;
saveas(gcf, 'visualizations/architectural_budget_distribution.png');
disp('     - Budget distribution plot saved.');

% --- 4. Generate Histogram and Boxplot for Each Feature ---
disp('Step 4: Generating histogram and boxplot for each feature...');
feature_names = features.Properties.VariableNames;

for i = 1:numel(feature_names)
    feature_name = feature_names{i};
    feature_data = features{:, i};
    
    % Clean feature name for filename
    clean_feature_name = matlab.lang.makeValidName(feature_name);
    
    % Create a single figure for both plots with a white background
    figure('Visible', 'off', 'Position', [100, 100, 1000, 400], 'Color', 'w');
    
    % Subplot 1: Histogram with Density Curve
    subplot(1, 2, 1);
    h = histogram(feature_data, 'Normalization', 'pdf');
    h.FaceColor = [0.678, 0.847, 0.902]; % Light blue color
    h.EdgeColor = [0.2, 0.2, 0.2];
    hold on;
    [f, xi] = ksdensity(feature_data);
    plot(xi, f, 'LineWidth', 1.5);
    hold off;
    title(['Histogram of ' strrep(feature_name, '_', ' ')]);
    xlabel(strrep(feature_name, '_', ' '));
    ylabel('Count');
    grid off;
    box on;
    
    % Subplot 2: Boxplot
    subplot(1, 2, 2);
    boxplot(feature_data, 'Orientation', 'horizontal');
    title(['Boxplot of ' strrep(feature_name, '_', ' ')]);
    xlabel(strrep(feature_name, '_', ' '));
    grid off;
    box on;
    
    % Save the combined figure
    saveas(gcf, ['visualizations/' clean_feature_name '_summary.png']);
end
disp('     - Individual feature plots saved.');

disp('--- All visualization processes complete. ---');
