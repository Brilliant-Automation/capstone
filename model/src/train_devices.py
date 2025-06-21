import argparse
import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


# Configure logging
def setup_logging(device_name):
    script_dir = os.path.dirname(__file__)
    log_file = os.path.join(script_dir, f"{device_name}_training.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(message)s",
        filemode="w"  # Overwrites the file for every run
    )
    logging.info("========== Training Log for Device: %s ==========\n", device_name)


# Define models
models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "param_grid": None
    },
    "Ridge Regression": {
        "model": Ridge(),
        "param_grid": {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(random_state=42),
        "param_grid": {
            "decisiontree__max_depth": [3, 5, 10, None],
            "decisiontree__min_samples_split": [2, 5, 10],
            "decisiontree__min_samples_leaf": [1, 2, 5]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "param_grid": {
            "randomforest__n_estimators": [50, 100, 200],
            "randomforest__max_depth": [5, 10, 15, None],
            "randomforest__min_samples_split": [2, 5],
            "randomforest__min_samples_leaf": [1, 2]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "param_grid": {
            "gradientboosting__n_estimators": [50, 100, 200],
            "gradientboosting__learning_rate": [0.01, 0.1, 0.2],
            "gradientboosting__max_depth": [3, 5, 7]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42),
        "param_grid": {
            "xgbregressor__n_estimators": [50, 100, 200],
            "xgbregressor__learning_rate": [0.01, 0.1, 0.2],
            "xgbregressor__max_depth": [3, 5, 7]
        }
    },
    "SVR": {
        "model": SVR(),
        "param_grid": {
            "svr__kernel": ["linear", "rbf"],
            "svr__C": [0.1, 1, 10],
            "svr__epsilon": [0.01, 0.1, 0.2]
        }
    }
}


# Preprocess data
def preprocess_data(df, input_features, target_features, drop_nan_targets=True):
    X = df[input_features]
    y = df[target_features]

    if drop_nan_targets:
        data = pd.concat([X, y], axis=1).dropna(subset=target_features)
        X = data[input_features]
        y = data[target_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Train and evaluate models
def train_and_evaluate(models, X_train, X_test, y_train, y_test, target_features):
    results = {}
    for model_name, config in models.items():
        logging.info("=== Training Model: %s ===", model_name)
        step_name = model_name.lower().replace(" ", "")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (step_name, config["model"])
        ])

        param_grid = {}
        if config["param_grid"]:
            param_grid = {f"{step_name}__{k.split('__')[-1]}": v for k, v in config["param_grid"].items()}

        model_results = {}
        for target in target_features:
            logging.info("    --- Target Feature: %s ---", target)
            if param_grid:
                grid_search = GridSearchCV(pipeline, param_grid, scoring="neg_mean_squared_error", cv=3)
                grid_search.fit(X_train, y_train[target])
                best_model = grid_search.best_estimator_
            else:
                pipeline.fit(X_train, y_train[target])
                best_model = pipeline

            # Predictions and Metrics
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_rmse = np.sqrt(mean_squared_error(y_train[target], y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test[target], y_test_pred))
            train_r2 = r2_score(y_train[target], y_train_pred)
            test_r2 = r2_score(y_test[target], y_test_pred)

            model_results[target] = {
                "Model Name": model_name,
                "Best Params": best_model,
                "Train RMSE": train_rmse,
                "Test RMSE": test_rmse,
                "Train R²": train_r2,
                "Test R²": test_r2
            }

            # Log metrics in a clean format
            logging.info(
                "        Train RMSE: %.4f | Test RMSE: %.4f | Train R²: %.4f | Test R²: %.4f",
                train_rmse, test_rmse, train_r2, test_r2
            )

        logging.info("=== Finished Training Model: %s ===\n", model_name)
        results[model_name] = model_results
    return results


# Save the overall best model per feature
def save_best_models_overall(results, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    best_models = {}

    for model_name, model_results in results.items():
        for target_feature, metrics in model_results.items():
            if target_feature not in best_models or metrics["Test R²"] > best_models[target_feature]["Test R²"]:
                best_models[target_feature] = {
                    "Model": metrics["Best Params"],
                    "Model Name": metrics["Model Name"],
                    "Test R²": metrics["Test R²"],
                    "Test RMSE": metrics["Test RMSE"],
                    "Train R²": metrics["Train R²"],
                    "Train RMSE": metrics["Train RMSE"]
                }

    # Save the best models
    for target_feature, best_model_info in best_models.items():
        model = best_model_info["Model"]
        file_name = f"{target_feature}_best_model.joblib"
        model_path = os.path.join(output_folder, file_name)
        joblib.dump(model, model_path)

        # Log the saving in a clean, grouped structure
        logging.info("### Best Model for Target: %s ###", target_feature)
        logging.info("    Model: %s", best_model_info["Model Name"])
        logging.info("    Train RMSE: %.4f | Train R²: %.4f", best_model_info["Train RMSE"],
                     best_model_info["Train R²"])
        logging.info("    Test RMSE: %.4f | Test R²: %.4f", best_model_info["Test RMSE"], best_model_info["Test R²"])
        logging.info("    Model Saved at: %s\n", model_path)


# Main function
def main(device):
    datasets = {
        "conveyor_belt": {
            "data_path": "Data/process/8#Belt Conveyer_merged.csv",
            "input_features": [
                'High-Frequency Acceleration_mean', 'High-Frequency Acceleration_min',
                'High-Frequency Acceleration_max',
                'High-Frequency Acceleration_median', 'High-Frequency Acceleration_std',
                'Low-Frequency Acceleration Z_mean',
                'Low-Frequency Acceleration Z_min', 'Low-Frequency Acceleration Z_max',
                'Low-Frequency Acceleration Z_median',
                'Low-Frequency Acceleration Z_std', 'Temperature_mean', 'Temperature_min', 'Temperature_max',
                'Temperature_median', 'Temperature_std', 'Vibration Velocity Z_mean', 'Vibration Velocity Z_min',
                'Vibration Velocity Z_max', 'Vibration Velocity Z_median', 'Vibration Velocity Z_std'
            ],
            "target_features": [
                'alignment_status', 'bearing_lubrication', 'crest_factor', 'electromagnetic_status', 'fit_condition',
                'kurtosis_opt', 'rms_10_25khz', 'rms_1_10khz', 'rotor_balance_status', 'rubbing_condition',
                'velocity_rms',
                'peak_value_opt'
            ]
        },
        "high_temp_fan": {
            "data_path": "Data/process/1#High-Temp Fan_merged.csv",
            "input_features": [
                'High-Frequency Acceleration_mean', 'High-Frequency Acceleration_min',
                'High-Frequency Acceleration_max', 'High-Frequency Acceleration_median',
                'High-Frequency Acceleration_std', 'Low-Frequency Acceleration Z_mean',
                'Low-Frequency Acceleration Z_min', 'Low-Frequency Acceleration Z_max',
                'Low-Frequency Acceleration Z_median', 'Low-Frequency Acceleration Z_std',
                'Temperature_mean', 'Temperature_min', 'Temperature_max', 'Temperature_median',
                'Temperature_std', 'Vibration Velocity Z_mean', 'Vibration Velocity Z_min',
                'Vibration Velocity Z_max', 'Vibration Velocity Z_median', 'Vibration Velocity Z_std'
            ],
            "target_features": [
                'alignment_status', 'bearing_lubrication', 'crest_factor',
                'electromagnetic_status', 'fit_condition', 'kurtosis_opt',
                'rms_10_25khz', 'rms_1_10khz', 'rotor_balance_status',
                'rubbing_condition', 'velocity_rms'
            ]
        },
        "tube_mill": {
            "data_path": "Data/process/Tube Mill_merged.csv",
            "input_features": [
                'High-Frequency Acceleration_mean', 'High-Frequency Acceleration_min',
                'High-Frequency Acceleration_max', 'High-Frequency Acceleration_median',
                'High-Frequency Acceleration_std', 'Low-Frequency Acceleration Z_mean',
                'Low-Frequency Acceleration Z_min', 'Low-Frequency Acceleration Z_max',
                'Low-Frequency Acceleration Z_median', 'Low-Frequency Acceleration Z_std',
                'Temperature_mean', 'Temperature_min', 'Temperature_max', 'Temperature_median',
                'Temperature_std', 'Vibration Velocity Z_mean', 'Vibration Velocity Z_min',
                'Vibration Velocity Z_max', 'Vibration Velocity Z_median', 'Vibration Velocity Z_std'
            ],
            "target_features": [
                'alignment_status', 'bearing_lubrication', 'crest_factor',
                'electromagnetic_status', 'fit_condition', 'kurtosis_opt',
                'rms_10_25khz', 'rms_1_10khz', 'rotor_balance_status',
                'rubbing_condition', 'velocity_rms',
                'peak_10_1000hz', 'peak_value_opt', 'rms_0_10hz', 'rms_10_100hz'
            ]
        }
        # Add definitions for other devices as needed
    }

    if device not in datasets:
        logging.error("Invalid device name '%s'. Choose from: %s", device, ", ".join(datasets.keys()))
        return

    # Setup logging
    setup_logging(device)

    dataset = datasets[device]
    df = pd.read_csv(dataset["data_path"])
    X_train, X_test, y_train, y_test = preprocess_data(
        df, dataset["input_features"], dataset["target_features"]
    )
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test, dataset["target_features"])
    save_best_models_overall(results, f"./{device}")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for a specific device.")
    parser.add_argument("device", type=str, help="Device name (e.g., 'conveyor_belt')")
    args = parser.parse_args()
    main(args.device)
