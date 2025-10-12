function [X_poly, feature_names_poly] = generatePolyFeatures(T_in, degree)
% generatePolyFeatures Creates polynomial and interaction features.
%   [X_poly, feature_names_poly] = generatePolyFeatures(T_in, degree)
%   takes a table T_in and a degree, and returns a matrix X_poly with
%   the original features plus all polynomial/interaction terms up to
%   the specified degree. It also returns the corresponding feature names.

% Convert table to a numeric array for easier computation
X = table2array(T_in);
original_names = T_in.Properties.VariableNames;
[~, n_features] = size(X);

% Start with the original features (degree 1)
X_poly = X;
feature_names_poly = original_names;

% Handle only degree 2 for simplicity and relevance to the main script
if degree == 2
    % Iterate through combinations of original features
    for i = 1:n_features
        % Create squared terms (e.g., year^2)
        new_feature_col_sq = X(:, i).^2;
        X_poly = [X_poly, new_feature_col_sq];
        new_name_sq = sprintf('%s^2', original_names{i});
        feature_names_poly = [feature_names_poly, new_name_sq];

        % Create interaction terms (e.g., year*num_storeys)
        for j = (i + 1):n_features % Use j=i+1 to avoid duplicates and self-multiplication
            
            % Create the new feature column
            new_feature_col_int = X(:, i) .* X(:, j);
            
            % Append the new column to the output matrix
            X_poly = [X_poly, new_feature_col_int];
            
            % Create and append the new feature name
            new_name_int = sprintf('%s*%s', original_names{i}, original_names{j});
            feature_names_poly = [feature_names_poly, new_name_int];
        end
    end
else
    error('This version of generatePolyFeatures only supports degree 2.');
end

fprintf('Generated %d polynomial features (up to degree %d).\n', size(X_poly, 2), degree);

end