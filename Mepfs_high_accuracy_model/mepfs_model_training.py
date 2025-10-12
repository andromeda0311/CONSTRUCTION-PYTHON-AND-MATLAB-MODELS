import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import lightgbm as lgb
import joblib
import os
from add_visualization import plot_correlation_matrix, plot_actual_vs_predicted

def remove_outliers(df, columns, z_threshold=3):
    """
    Remove outliers using Z-score method.
    """
    df_cleaned = df.copy()
    initial_rows = len(df_cleaned)
    
    for col in columns:
        z_scores = np.abs(stats.zscore(df_cleaned[col]))
        df_cleaned = df_cleaned[z_scores < z_threshold]
    
    removed_rows = initial_rows - len(df_cleaned)
    print(f"Removed {removed_rows} outlier rows (Z-score threshold: {z_threshold})")
    return df_cleaned

def engineer_features(df):
    """
    Create additional features to improve model performance.
    """
    df_featured = df.copy()
    
    # Identify feature columns (exclude Year and Budget)
    feature_columns = [col for col in df_featured.columns if col not in ['Year', 'Budget']]
    
    # 1. Total cost of all MEPFS components
    df_featured['Total_Features_Cost'] = df_featured[feature_columns].sum(axis=1)
    
    # 2. Count of non-zero MEPFS components
    df_featured['Non_Zero_Features'] = (df_featured[feature_columns] > 0).sum(axis=1)
    
    # 3. Average cost per non-zero component
    epsilon = 1e-6
    df_featured['Avg_Cost_Per_Feature'] = df_featured['Total_Features_Cost'] / (df_featured['Non_Zero_Features'] + epsilon)
    
    # 4. Standard deviation of feature costs
    df_featured['Std_Features_Cost'] = df_featured[feature_columns].std(axis=1)
    
    # 5. Max feature cost
    df_featured['Max_Feature_Cost'] = df_featured[feature_columns].max(axis=1)
    
    print(f"Created {5} new engineered features")
    return df_featured

def train_budget_prediction_model(data_filepath, model_output_dir, visualization_dir):
    """
    Loads preprocessed data, applies feature engineering, trains a LightGBM model
    with hyperparameter tuning using RandomizedSearchCV, evaluates it, and saves the trained model.
    """
    # --- Load Data ---
    df = pd.read_csv(data_filepath)
    print("Loaded preprocessed data successfully.")
    print(f"Initial data shape: {df.shape}")

    # --- Remove Outliers ---
    feature_columns = [col for col in df.columns if col not in ['Year', 'Project']]
    df = remove_outliers(df, feature_columns, z_threshold=3)
    print(f"Data shape after outlier removal: {df.shape}")

    # --- Feature Engineering ---
    print("\nApplying feature engineering...")
    df = engineer_features(df)

    # --- Generate Correlation Matrix ---
    plot_correlation_matrix(df.drop('Year', axis=1, errors='ignore'), visualization_dir)

    # --- Apply Log Transformation to Target ---
    print("\nApplying log transformation to 'Budget'...")
    df['Budget'] = np.log1p(df['Budget'])

    # --- Prepare Features (X) and Target (y) ---
    X = df.drop('Budget', axis=1)
    y = df['Budget']
    
    if 'Year' in X.columns:
        X = X.drop('Year', axis=1)
        
    print(f"\nTarget variable: Budget (log-transformed)")
    print(f"Number of features: {len(X.columns)}")

    # --- Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # --- Feature Scaling ---
    print("\nApplying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train the Model with Hyperparameter Tuning ---
    print("\nTraining LightGBM model with hyperparameter tuning...")
    
    # Define the estimator
    estimator = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
    
    # Expanded parameter grid for RandomizedSearchCV
    param_distributions = {
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.15],
        'n_estimators': [500, 1000, 1500, 2000],
        'num_leaves': [20, 31, 40, 50, 70],
        'max_depth': [5, 7, 10, 15, -1],
        'min_child_samples': [10, 20, 30, 50],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }
    
    # Use RandomizedSearchCV for efficiency
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter settings sampled
        scoring='r2',
        n_jobs=-1,
        cv=5,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train_scaled, y_train)
    print("Hyperparameter tuning complete.")
    
    print(f"\nBest R² Score from CV: {random_search.best_score_:.4f}")
    print(f"Best Parameters: {random_search.best_params_}")
    
    # Use the best estimator
    best_model = random_search.best_estimator_

    # --- Evaluate the Model ---
    print("\nEvaluating model performance...")
    
    # Training set evaluation
    y_train_pred_log = best_model.predict(X_train_scaled)
    y_train_pred = np.expm1(y_train_pred_log)
    y_train_actual = np.expm1(y_train)
    
    train_r2 = r2_score(y_train_actual, y_train_pred)
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Training RMSE: {train_rmse:.2f}")

    # Testing set evaluation
    y_test_pred_log = best_model.predict(X_test_scaled)
    y_test_pred = np.expm1(y_test_pred_log)
    y_test_actual = np.expm1(y_test)
    
    test_r2 = r2_score(y_test_actual, y_test_pred)
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    
    print(f"\nTesting R²: {test_r2:.4f}")
    print(f"Testing MAE: {test_mae:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")

    # --- Cross-Validation Score ---
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"CV R² Scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # --- Generate Visualizations ---
    plot_actual_vs_predicted(y_train_actual, y_train_pred, visualization_dir, data_set_name='Training')
    plot_actual_vs_predicted(y_test_actual, y_test_pred, visualization_dir, data_set_name='Testing')
    
    # --- Save the Trained Model and Scaler ---
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        
    model_filepath = os.path.join(model_output_dir, 'mepfs_budget_model_lgbm_tuned.joblib')
    scaler_filepath = os.path.join(model_output_dir, 'mepfs_scaler.joblib')
    
    joblib.dump(best_model, model_filepath)
    joblib.dump(scaler, scaler_filepath)
    
    print(f"\nTrained model saved to {model_filepath}")
    print(f"Feature scaler saved to {scaler_filepath}")
    
    # --- Feature Importance ---
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importance
    importance_filepath = os.path.join(model_output_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_filepath, index=False)
    print(f"Feature importance saved to {importance_filepath}")

if __name__ == "__main__":
    PREPROCESSED_DATA_PATH = 'mepfs_preprocessed_data.csv'
    MODEL_OUTPUT_DIR = 'models'
    VISUALIZATION_DIR = 'visualizations'
    
    train_budget_prediction_model(PREPROCESSED_DATA_PATH, MODEL_OUTPUT_DIR, VISUALIZATION_DIR)
