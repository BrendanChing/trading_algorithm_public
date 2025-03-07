import sqlite3
import pandas as pd
import numpy as np
import logging
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Database path
DB_PATH = "stock_data.db"
FEATURES_TABLE = "features"
SCALING_PARAMS_TABLE = "scaling_params"
BATCH_SIZE = 10000  # Number of rows to update per batch commit

# List of features to normalize (newly added features)
FEATURES_TO_NORMALIZE = [
    'dist_min_ma_26',
    'dist_max_ma_26',
    'dist_min_ma_260',
    'dist_max_ma_260',
    'dist_ma_step1',
    'dist_ma_step26',
    'std_dev_open_price_16',
    'rsi_14'
]

def add_normalized_columns(conn, table, features):
    """
    Adds new columns for normalized features with a '_z' suffix if they do not already exist.
    """
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        existing_columns = set(row[1] for row in cur.fetchall())

        # Determine which normalized columns need to be added
        normalized_cols = [f"{feat}_z" for feat in features]
        needed_cols = [col for col in normalized_cols if col not in existing_columns]

        for col in needed_cols:
            logging.info(f"Adding normalized column '{col}' to table '{table}'...")
            alter_sql = f"ALTER TABLE {table} ADD COLUMN {col} REAL"
            cur.execute(alter_sql)

        conn.commit()
        cur.close()

        if needed_cols:
            logging.info(f"Added normalized columns: {', '.join(needed_cols)}")
        else:
            logging.info("No new normalized columns needed to be added.")

    except sqlite3.Error as e:
        logging.error(f"Failed to add normalized feature columns: {e}")
        conn.rollback()
        raise

def calculate_scaling_parameters(conn, table, features):
    """
    Calculates mean and standard deviation for each feature and stores them in scaling_params table.
    Null values are ignored in calculations.
    """
    try:
        logging.info("Calculating scaling parameters (mean and std) for each feature...")

        # Connect to the database
        cur = conn.cursor()

        # Iterate over each feature
        for feature in features:
            # Calculate mean and std, ignoring nulls
            query = f"""
                SELECT AVG({feature}) as mean, 
                       AVG(({feature} - (SELECT AVG({feature}) FROM {table})) * ({feature} - (SELECT AVG({feature}) FROM {table}))) as variance
                FROM {table}
                WHERE {feature} IS NOT NULL
            """
            cur.execute(query)
            result = cur.fetchone()
            mean = result[0]
            variance = result[1]
            std = np.sqrt(variance) if variance is not None else None

            if mean is None or std is None:
                logging.warning(f"Feature '{feature}' has insufficient data for scaling.")
                continue

            # Insert or update scaling_params table
            cur.execute(f"""
                INSERT INTO {SCALING_PARAMS_TABLE} (feature, mean, std)
                VALUES (?, ?, ?)
                ON CONFLICT(feature) DO UPDATE SET
                    mean=excluded.mean,
                    std=excluded.std
            """, (feature, mean, std))

            logging.info(f"Feature '{feature}': mean = {mean}, std = {std}")

        # Commit the scaling parameters
        conn.commit()
        cur.close()

    except sqlite3.Error as e:
        logging.error(f"Failed to calculate scaling parameters: {e}")
        conn.rollback()
        raise

def normalize_features(conn, table, features, batch_size=10000):
    """
    Normalizes the specified features using z-scaling and updates the normalized columns.
    """
    try:
        logging.info("Starting normalization of features...")

        # Load scaling parameters from scaling_params table
        scaling_df = pd.read_sql_query(f"SELECT * FROM {SCALING_PARAMS_TABLE} WHERE feature IN ({','.join(['?']*len(features))})", conn, params=features)
        scaling_params = scaling_df.set_index('feature').to_dict(orient='index')

        # Add normalized columns if they don't exist
        add_normalized_columns(conn, table, features)

        # Prepare for batch processing
        total_features = len(features)
        normalized_features = [f"{feat}_z" for feat in features]

        # Iterate over features to normalize
        for feat, norm_feat in zip(features, normalized_features):
            if feat not in scaling_params:
                logging.warning(f"Scaling parameters for feature '{feat}' not found. Skipping normalization for this feature.")
                continue

            mean = scaling_params[feat]['mean']
            std = scaling_params[feat]['std']

            if std == 0:
                logging.warning(f"Standard deviation for feature '{feat}' is zero. Skipping normalization for this feature.")
                continue

            logging.info(f"Normalizing feature '{feat}' into '{norm_feat}' with mean={mean} and std={std}.")

            # Update the normalized column using SQL
            update_sql = f"""
                UPDATE {table}
                SET {norm_feat} = ({feat} - ?) / ?
                WHERE {feat} IS NOT NULL
            """
            conn.execute("BEGIN TRANSACTION;")
            conn.execute(update_sql, (mean, std))
            conn.commit()
            logging.info(f"Normalized feature '{feat}' updated successfully.")

        logging.info("All specified features have been normalized.")

    except sqlite3.Error as e:
        logging.error(f"Failed to normalize features: {e}")
        conn.rollback()
        raise

def verify_normalization(conn, table, features, sample_size=5):
    """
    Verifies that the normalization was successful by displaying a sample of records.
    """
    try:
        normalized_features = [f"{feat}_z" for feat in features]
        select_cols = ['symbol', 'date', 'time'] + features + normalized_features

        query = f"""
            SELECT {', '.join(select_cols)}
            FROM {table}
            WHERE {' OR '.join([f"{feat}_z IS NOT NULL" for feat in features])}
            LIMIT ?
        """
        df_sample = pd.read_sql_query(query, conn, params=(sample_size,))

        if not df_sample.empty:
            logging.info("Sample of records with original and normalized features:")
            logging.info("\n" + tabulate(df_sample, headers='keys', tablefmt='psql'))
        else:
            logging.warning("No records found with normalized features. Please check the normalization process.")

    except sqlite3.Error as e:
        logging.error(f"Failed to verify normalization: {e}")
        raise

def main():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DB_PATH)
        logging.info(f"Connected to database at '{DB_PATH}'.")

        # Step 1: Calculate scaling parameters and store them in scaling_params table
        calculate_scaling_parameters(conn, FEATURES_TABLE, FEATURES_TO_NORMALIZE)

        # Step 2: Normalize features and update the features table
        normalize_features(conn, FEATURES_TABLE, FEATURES_TO_NORMALIZE, BATCH_SIZE)

        # Step 3: Verify the normalization by displaying a sample of records
        verify_normalization(conn, FEATURES_TABLE, FEATURES_TO_NORMALIZE, sample_size=5)

    except Exception as e:
        logging.error(f"An error occurred during normalization: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
