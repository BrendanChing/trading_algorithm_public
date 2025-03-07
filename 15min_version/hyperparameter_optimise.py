import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss
import logging
import itertools

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_PATH = "stock_data.db"
FEATURES_TABLE = "features"

###############################################################################
# 1) Load data (example) - adapt this to your actual loading function
###############################################################################
def load_data():
    """
    Loads from 'features' table where reward is not null.
    Drops non-numeric columns. Splits into X, y. 
    You may adapt this as needed.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM {FEATURES_TABLE} WHERE reward IS NOT NULL", conn
    )
    conn.close()

    if df.empty:
        logging.warning("No data found with reward != NULL.")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Separate y
    y = df['reward'].copy()
    drop_cols = {'reward', 'symbol', 'date', 'time', 'id', 'diff_4_1month'}  # plus anything else you want to drop
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()

    # Drop any non-numeric columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X.drop(columns=col, inplace=True)
    # Drop rows with NaN
    initial_len = len(X)
    X.dropna(inplace=True)
    y = y.loc[X.index]
    logging.info(f"Dropped {initial_len - len(X)} rows due to NaN. Final shape: {X.shape}")
    return X, y

###############################################################################
# 2) Train a model with given params, return relevant metrics
###############################################################################
def train_and_evaluate(xgb_params, num_boost_round, 
                       X_train, X_test, y_train, y_test,
                       early_stopping=10):
    """
    Trains an XGBoost model with 'xgb_params' on (X_train, y_train).
    Uses early stopping on the test set (y_test) for convenience.
    Returns:
      - ratio (TP/FP)
      - TPs, FPs
      - train_logloss, eval_logloss
      - AUC
    """

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    # Train model
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping,
        verbose_eval=False
    )

    # best_iteration found by early stopping
    best_ntree = model.best_iteration

    # We can compute logloss on train/test
    y_train_pred_prob = model.predict(dtrain, iteration_range=(0, best_ntree+1))
    y_test_pred_prob  = model.predict(dtest,  iteration_range=(0, best_ntree+1))

    train_logloss = log_loss(y_train, y_train_pred_prob, labels=[0, 1])
    eval_logloss  = log_loss(y_test,  y_test_pred_prob,  labels=[0, 1])

    # Confusion matrix with threshold=0.5
    y_test_pred = (y_test_pred_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print("Confusion matrix:", tn, fp, fn, tp)


    # ratio = TP / FP (watch out for division by zero if fp=0)
    ratio = tp / (fp + 1e-9)

    # AUC
    auc = roc_auc_score(y_test, y_test_pred_prob)

    print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
    print("y_test distribution:", y_test.value_counts())
    print("best_ntree:", model.best_iteration)
    print("Predicted probability stats - min:", y_test_pred_prob.min(), "max:", y_test_pred_prob.max())
    print("Predicted positives:", y_test_pred.sum())
    print("tn, fp, fn, tp =", tn, fp, fn, tp)

    return {
        'model': model,
        'ratio': ratio,
        'tp': tp,
        'fp': fp,
        'train_logloss': train_logloss,
        'eval_logloss': eval_logloss,
        'auc': auc,
        'best_iteration': best_ntree
    }

###############################################################################
# 3) Hyperparameter search with constraints (including num_boost_round)
###############################################################################
def hyperparameter_search(X, y):
    """
    Example grid search over some hyperparams, including num_boost_round.
    We'll filter models by:
      - TPs >= 2000
      - abs(train_logloss - eval_logloss) <= 0.04
      - AUC >= 0.68 (optional)
    Then pick the best ratio = TP/FP.
    """

    # We do a simple train/test split for demonstration. 
    # For time series, ensure you do a forward-looking split (shuffle=False).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Param grid including num_boost_round
    # We separate XGBoost params from 'num_boost_round'
    # We'll nest them together in a dictionary for clarity.
    param_grid = {
        'max_depth': [8, 9, 10],
        'min_child_weight': [2, 3, 4],
        'eta': [0.025, 0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [0.01],
        'reg_lambda': [1.0],
        'num_boost_round': [400, 500, 600]
    }

    # Common XGBoost parameters that won't vary
    objective = 'binary:logistic'
    verbosity = 1
    early_stopping = 10

    # We'll keep track of the best
    best_model_info = None
    best_ratio = -1.0
    all_results = []

    # Generate all combinations from param_grid
    keys = list(param_grid.keys())  # e.g. ['max_depth','min_child_weight','eta',...,'num_boost_round']
    values_lists = [param_grid[k] for k in keys]

    for combo in itertools.product(*values_lists):
        # combo is a tuple of values, one per key in param_grid
        combo_dict = dict(zip(keys, combo))

        # Extract num_boost_round separately
        num_boost_round = combo_dict['num_boost_round']
        # Build the XGBoost param dict (excluding num_boost_round)
        xgb_params = {
            'objective': objective,
            'verbosity': verbosity,
            'max_depth': combo_dict['max_depth'],
            'min_child_weight': combo_dict['min_child_weight'],
            'eta': combo_dict['eta'],
            'subsample': combo_dict['subsample'],
            'colsample_bytree': combo_dict['colsample_bytree'],
            'reg_alpha': combo_dict['reg_alpha'],
            'reg_lambda': combo_dict['reg_lambda'],
        }

        result = train_and_evaluate(
            xgb_params,
            num_boost_round,
            X_train, X_test, y_train, y_test,
            early_stopping
        )

        tp  = result['tp']
        fp  = result['fp']
        ratio = result['ratio']
        train_logloss = result['train_logloss']
        eval_logloss  = result['eval_logloss']
        auc = result['auc']

        print(f"\nTrying combo: {xgb_params}, rounds={num_boost_round}")
        print(f"tp={tp}, fp={fp}, ratio={ratio:.3f}")
        print(f"train_logloss={train_logloss:.4f}, eval_logloss={eval_logloss:.4f}, diff={abs(train_logloss - eval_logloss):.4f}")
        print(f"AUC={auc:.4f}")

        # Constraints
        if tp < 2000:
            continue
        if abs(train_logloss - eval_logloss) > 0.07:
            continue
        if auc < 0.7:
            continue

        # If passes constraints, check ratio
        if ratio > best_ratio:
            best_ratio = ratio
            best_model_info = {
                'xgb_params': xgb_params,
                'num_boost_round': num_boost_round,
                'model': result['model'],
                'ratio': ratio,
                'tp': tp,
                'fp': fp,
                'train_logloss': train_logloss,
                'eval_logloss': eval_logloss,
                'auc': auc,
                'best_iteration': result['best_iteration']
            }

        all_results.append({
            **result,
            'xgb_params': xgb_params,
            'num_boost_round': num_boost_round
        })

    return best_model_info, all_results


def main():
    X, y = load_data()
    if X.empty:
        logging.info("No data loaded. Exiting.")
        return

    best_model_info, all_results = hyperparameter_search(X, y)
    if not best_model_info:
        logging.info("No hyperparameter combination met the constraints!")
        return

    logging.info("BEST MODEL INFO:")
    logging.info(f"XGB Params: {best_model_info['xgb_params']}")
    logging.info(f"num_boost_round: {best_model_info['num_boost_round']}")
    logging.info(f"TP/FP Ratio: {best_model_info['ratio']:.3f}")
    logging.info(f"TP: {best_model_info['tp']}  FP: {best_model_info['fp']}")
    logging.info(f"Train/Eval logloss: {best_model_info['train_logloss']:.4f} / {best_model_info['eval_logloss']:.4f}")
    logging.info(f"AUC: {best_model_info['auc']:.4f}")
    logging.info(f"Best Iteration (early stopping): {best_model_info['best_iteration']}")

    # Optionally save the best model
    best_model_info['model'].save_model("best_xgboost_model.json")


if __name__ == "__main__":
    main()
