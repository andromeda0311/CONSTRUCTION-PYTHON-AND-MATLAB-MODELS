function [X_poly, feature_names_poly] = generatePolyFeatures(T_in, degree)
% generatePolyFeatures Creates polynomial and interaction features.
%   [X_poly, feature_names_poly] = generatePolyFeatures(T_in, degree)
%   takes a table T_in and a degree, and returns a matrix X_poly with
%   the original features plus all polynomial/interaction terms up to
%   the specified degree. It also returns the corresponding feature names.

% Convert table to a numeric array for easier computation
X = table2array(T_in);
original_names = T_in.Properties.VariableNames;
[n_samples, n_features] = size(X);

% Start with the original features (degree 1)
X_poly = X;
feature_names_poly = original_names;

% Generate features for degrees 2 up to the specified degree
for d = 2:degree
    % Iterate through combinations of original features
    for i = 1:n_features
        for j = i:n_features % Use j=i to avoid duplicate interaction terms (e.g., feat1*feat2 and feat2*feat1)
            
            % Create the new feature column
            new_feature_col = X(:, i) .* X(:, j);
            
            % Append the new column to the output matrix
            X_poly = [X_poly, new_feature_col];
            
            % Create and append the new feature name
            if i == j
                % This is a pure power term (e.g., year^2)
                new_name = sprintf('%s^%d', original_names{i}, d);
            else
                % This is an interaction term (e.g., year*num_storeys)
                new_name = sprintf('%s*%s', original_names{i}, original_names{j});
            end
            feature_names_poly = [feature_names_poly, new_name];
        end
    end
end

fprintf('Generated %d polynomial features (up to degree %d).\n', size(X_poly, 2), degree);

end