import sqlite3
import pandas as pd
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Database path
DB_PATH = "stock_data.db"
SRC_TABLE = "historical_15min"
SCALING_PARAMS_TABLE = "scaling_params"

# Columns to exclude from normalization
EXCLUDED_COLUMNS = {"ID", "high_price", "low_price", "close_price", "reward"}

def create_scaling_params_table(conn, table_name=SCALING_PARAMS_TABLE):
    """
    Creates the 'scaling_params' table to store mean and std for each feature.
    Drops the table if it exists.
    """
    try:
        logging.info(f"Dropping table '{table_name}' if it exists...")
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()

        # Create table
        create_sql = f"""
            CREATE TABLE {table_name} (
                feature TEXT PRIMARY KEY,
                mean REAL,
                std REAL
            )
        """
        logging.info(f"Creating table '{table_name}' with schema:\n{create_sql}")
        cur.execute(create_sql)
        conn.commit()
        cur.close()
        logging.info(f"Table '{table_name}' created successfully.")
    except Exception as e:
        logging.error(f"Failed to create '{table_name}': {e}")
        raise

def load_historical_data(conn, table_name=SRC_TABLE):
    """
    Loads all relevant columns from the 'historical_15min' table into a DataFrame.
    Excludes specified columns.
    """
    try:
        logging.info(f"Loading data from '{table_name}'...")
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        logging.info(f"Loaded {len(df)} rows with columns {list(df.columns)}.")

        # Identify columns to include (exclude specified columns)
        included_columns = [col for col in df.columns if col not in EXCLUDED_COLUMNS]

        # Further filter to include only numeric columns
        numeric_columns = df[included_columns].select_dtypes(include=[np.number]).columns.tolist()

        logging.info(f"Columns selected for scaling: {numeric_columns}")
        return df[numeric_columns]
    except Exception as e:
        logging.error(f"Failed to load data from '{table_name}': {e}")
        raise

def calculate_scaling_parameters(df):
    """
    Calculates mean and standard deviation for each column in the DataFrame.
    Returns a DataFrame with scaling parameters.
    """
    try:
        logging.info("Calculating scaling parameters...")
        scaling_params = df.agg(['mean', 'std']).transpose().reset_index()
        scaling_params.columns = ['feature', 'mean', 'std']
        logging.info("Scaling parameters calculated successfully.")
        logging.debug(f"Scaling Parameters:\n{scaling_params}")
        return scaling_params
    except Exception as e:
        logging.error(f"Failed to calculate scaling parameters: {e}")
        raise

def insert_scaling_params(conn, scaling_params_df, table_name=SCALING_PARAMS_TABLE):
    """
    Inserts scaling parameters into the 'scaling_params' table.
    Uses UPSERT to handle existing entries.
    """
    try:
        logging.info(f"Inserting scaling parameters into '{table_name}'...")
        cur = conn.cursor()

        for _, row in scaling_params_df.iterrows():
            feature = row['feature']
            mean = row['mean']
            std = row['std']
            logging.info(f"Inserting scaling params for feature '{feature}': Mean={mean}, Std={std}")

            # UPSERT statement to handle duplicates
            upsert_sql = f"""
                INSERT INTO {table_name} (feature, mean, std)
                VALUES (?, ?, ?)
                ON CONFLICT(feature) DO UPDATE SET
                    mean=excluded.mean,
                    std=excluded.std;
            """
            cur.execute(upsert_sql, (feature, mean, std))

        conn.commit()
        cur.close()
        logging.info(f"Scaling parameters inserted/updated successfully in '{table_name}'.")
    except Exception as e:
        logging.error(f"Failed to insert scaling parameters into '{table_name}': {e}")
        conn.rollback()
        raise

def verify_scaling_params(conn, table_name=SCALING_PARAMS_TABLE, sample_size=5):
    """
    Verifies that scaling parameters have been inserted correctly by displaying a sample.
    """
    try:
        logging.info(f"Verifying entries in '{table_name}'...")
        query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
        scaling_params = pd.read_sql(query, conn)
        logging.info("Sample of scaling parameters:")
        logging.info("\n" + scaling_params.to_string(index=False))
    except Exception as e:
        logging.error(f"Failed to verify scaling parameters: {e}")
        raise

def main():
    # Connect to the SQLite database
    try:
        conn = sqlite3.connect(DB_PATH)
        logging.info(f"Connected to database at '{DB_PATH}'.")
    except Exception as e:
        logging.error(f"Failed to connect to database at '{DB_PATH}': {e}")
        return

    try:
        # Step 1: Create scaling_params table
        create_scaling_params_table(conn, SCALING_PARAMS_TABLE)

        # Step 2: Load historical_15min data
        df = load_historical_data(conn, SRC_TABLE)

        if df.empty:
            logging.warning(f"No data found in '{SRC_TABLE}'. Exiting.")
            return

        # Step 3: Calculate scaling parameters
        scaling_params_df = calculate_scaling_parameters(df)

        if scaling_params_df.empty:
            logging.warning("No scaling parameters calculated. Exiting.")
            return

        # Step 4: Insert scaling parameters into scaling_params table
        insert_scaling_params(conn, scaling_params_df, SCALING_PARAMS_TABLE)

        # Step 5: Verify insertion
        verify_scaling_params(conn, SCALING_PARAMS_TABLE, sample_size=5)

        logging.info("Scaling parameters calculation and insertion completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during scaling parameters processing: {e}")
    finally:
        conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
