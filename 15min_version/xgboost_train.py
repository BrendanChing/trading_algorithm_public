import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import logging

# Database path
db_path = 'stock_data.db'
features_table = 'features'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_15min_data():
    """
    Loads data from the 'features' table, using 'reward' as the target,
    and all other numeric columns (except symbol, date, time, reward) as features.
    Drops rows where reward is NULL.
    """
    logging.info("Connecting to the database and loading data...")
    conn = sqlite3.connect(db_path)

    # Load entire table where reward is not null
    df = pd.read_sql_query(f"SELECT * FROM {features_table} WHERE reward IS NOT NULL", conn)
    conn.close()

    if df.empty:
        logging.warning("No rows found where reward is NOT NULL in 'features' table.")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Separate target (reward) from the rest
    y = df['reward'].copy()

    # Drop non-feature columns: symbol, date, time, reward
    drop_cols = {'symbol', 'date', 'time', 'reward', 'id', 'diff_4_1month', 'high_price', 'low_price',
                  'close_price', 'volume', 'actual_open_price', 'symbol_numeric', 'distance_to_min_ma_26',
                    'distance_to_max_ma_26', 'distance_to_min_ma_260', 'distance_to_max_ma_260', 'distance_to_ma_5_short',
                      'distance_to_ma_5_daily', 'std_dev_16', 'rsi', 'dist_min_ma_26', 'dist_max_ma_26',
                      'dist_min_ma_260', 'dist_max_ma_260', 'dist_ma_step1', 'dist_ma_step26', 'std_dev_open_price_16', 'rsi_14'}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()

    # Optionally, check for any non-numeric columns and drop them or convert them
    # We'll just drop them to be safe
    non_numeric = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric:
        logging.info(f"Dropping non-numeric columns: {non_numeric}")
        X.drop(columns=non_numeric, inplace=True)

    # Drop rows with NaN in either X or y
    logging.info("Dropping rows with missing values...")
    initial_len = len(X)
    X.dropna(inplace=True)
    y = y.loc[X.index]  # ensure alignment
    logging.info(f"Dropped {initial_len - len(X)} rows due to NaNs.")

    return X, y

def train_xgboost_15min():
    """
    Trains an XGBoost model on the 15-minute features.
    Saves the model as 'xgboost_15min.json'.
    """
    # 1) Load data
    X, y = load_15min_data()
    if X.empty or y.empty:
        print("No data available for training. Exiting.")
        return

    # 2) Train/test split
    logging.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 4) Define XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 9,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'verbosity': 1
    }

    # 5) Train model
    logging.info("Starting XGBoost training...")
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=800, evals=evals, early_stopping_rounds=10)

    # 6) Predictions and metrics
    logging.info("Making predictions on test set...")
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob >= 0.60).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # 7) Save model
    model_name = 'xgboost_15min_more.json'
    logging.info(f"Saving model to {model_name}...")
    model.save_model(model_name)

    # 8) Feature Importance
    logging.info("Plotting feature importance...")
    feature_importance = model.get_score(importance_type='weight')
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_names = [k for k, v in sorted_importance]
    importance_values = [v for k, v in sorted_importance]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance_values)
    plt.xlabel('Feature Importance Score')
    plt.title('XGBoost Feature Importance (15min Model)')
    plt.gca().invert_yaxis()  # Most important at the top
    plt.show()

if __name__ == "__main__":
    train_xgboost_15min()
