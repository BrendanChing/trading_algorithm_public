import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Database path
DB_PATH = "stock_data.db"
SRC_TABLE = "features"  # Normalization is performed on the 'features' table
SCALING_PARAMS_TABLE = "scaling_params"
BATCH_SIZE = 10000  # Number of rows to update per batch commit

# Columns to exclude from normalization
EXCLUDED_COLUMNS = {"ID", "high_price", "low_price", "close_price"}

# Columns to normalize and their corresponding new columns
NORMALIZE_COLUMNS = {
    "actual_open_price": "open_price",  # 'actual_open_price' will be normalized and stored in 'open_price'
    "symbol_numeric": "symbol_numeric_normalized"
}

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


def rename_column(conn, table_name, old_name, new_name):
    """
    Renames a column in the specified table.
    """
    try:
        logging.info(f"Renaming column '{old_name}' to '{new_name}' in table '{table_name}'...")
        cur = conn.cursor()
        rename_sql = f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name};"
        cur.execute(rename_sql)
        conn.commit()
        cur.close()
        logging.info(f"Column '{old_name}' renamed to '{new_name}' successfully.")
    except sqlite3.OperationalError as e:
        logging.error(f"Failed to rename column '{old_name}' to '{new_name}': {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during column rename: {e}")
        raise


def add_new_columns(conn, table_name, columns):
    """
    Adds new columns to the specified table.
    :param columns: Dictionary where key is column name and value is data type
    """
    try:
        cur = conn.cursor()
        for col_name, col_type in columns.items():
            logging.info(f"Adding column '{col_name}' of type '{col_type}' to table '{table_name}'...")
            alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type};"
            cur.execute(alter_sql)
        conn.commit()
        cur.close()
        logging.info(f"All specified columns added to '{table_name}' successfully.")
    except sqlite3.OperationalError as e:
        logging.error(f"Failed to add columns to '{table_name}': {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during adding columns: {e}")
        raise


def load_features_table_into_df(conn, table_name=SRC_TABLE):
    """
    Loads all columns from the 'features' table into a DataFrame.
    Includes the implicit 'rowid' for matching during updates.
    """
    logging.info(f"Loading data from '{table_name}' into DataFrame...")
    query = f"SELECT rowid, * FROM {table_name}"
    df = pd.read_sql(query, conn)
    logging.info(f"Loaded {len(df)} rows with columns {list(df.columns)}.")
    return df


def standard_scale_selected_columns(df, columns_to_normalize):
    """
    Performs standard scaling on selected columns.
    :param df: DataFrame containing the data
    :param columns_to_normalize: Dictionary where key is original column name and value is new column name
    :return: Scaled DataFrame and scaling parameters DataFrame
    """
    scaler = StandardScaler()
    scaling_params_list = []

    for original_col, new_col in columns_to_normalize.items():
        if original_col not in df.columns:
            logging.warning(f"Column '{original_col}' not found in DataFrame. Skipping normalization for this column.")
            continue

        logging.info(f"Normalizing column '{original_col}' and storing in '{new_col}'...")
        # Handle non-numeric data
        if not pd.api.types.is_numeric_dtype(df[original_col]):
            logging.warning(f"Column '{original_col}' is not numeric. Skipping normalization for this column.")
            continue

        # Fit scaler and transform
        mean = df[original_col].mean()
        std = df[original_col].std()
        if std == 0:
            logging.warning(f"Standard deviation for column '{original_col}' is zero. Skipping normalization for this column.")
            df[new_col] = 0
            scaling_params_list.append({'feature': new_col, 'mean': mean, 'std': std})
            continue

        df[new_col] = scaler.fit_transform(df[[original_col]])
        scaling_params_list.append({'feature': new_col, 'mean': mean, 'std': std})
        logging.info(f"Column '{original_col}' normalized. Mean: {mean}, Std: {std}")

    # Convert list of dicts to DataFrame
    scaling_params = pd.DataFrame(scaling_params_list)
    logging.info(f"Scaling parameters collected for {len(scaling_params)} features.")
    logging.debug(f"Scaling Parameters DataFrame:\n{scaling_params}")
    return df, scaling_params


def update_features_table(conn, df, columns_to_update, batch_size=BATCH_SIZE):
    """
    Updates the 'features' table with normalized values in batches.
    :param conn: SQLite connection object
    :param df: DataFrame containing the data
    :param columns_to_update: List of column names to update
    :param batch_size: Number of records to update per batch
    """
    if not columns_to_update:
        logging.info("No columns to update in the 'features' table.")
        return

    logging.info(f"Updating columns {columns_to_update} in the 'features' table...")
    cur = conn.cursor()

    total_rows = len(df)
    logging.info(f"Total rows to update: {total_rows}")
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch_df = df.iloc[start:end]

        # Prepare the UPDATE statements
        update_tuples = []
        for _, row in batch_df.iterrows():
            update_values = tuple(row[col] for col in columns_to_update)
            rowid = row['rowid']
            update_tuples.append((*update_values, rowid))

        # Construct the SQL statement dynamically based on columns to update
        set_clause = ", ".join([f"{col} = ?" for col in columns_to_update])
        sql = f"UPDATE {SRC_TABLE} SET {set_clause} WHERE rowid = ?"

        try:
            cur.executemany(sql, update_tuples)
            conn.commit()
            logging.info(f"Updated rows {start + 1} to {end}.")
        except Exception as e:
            logging.error(f"Failed to update rows {start + 1} to {end}: {e}")
            conn.rollback()
            raise

    cur.close()
    logging.info("Completed updating the 'features' table with normalized values.")


def insert_scaling_params(conn, scaling_params_df, table_name=SCALING_PARAMS_TABLE):
    """
    Inserts scaling parameters into the 'scaling_params' table.
    """
    if scaling_params_df.empty:
        logging.info("No scaling parameters to insert.")
        return

    cols = list(scaling_params_df.columns)
    col_placeholders = ", ".join(["?"] * len(cols))
    insert_sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({col_placeholders})"

    records = list(scaling_params_df.itertuples(index=False, name=None))

    cur = conn.cursor()
    try:
        cur.executemany(insert_sql, records)
        conn.commit()
        logging.info(f"Inserted scaling parameters for {len(records)} features into '{table_name}'.")
    except Exception as e:
        logging.error(f"Failed to insert scaling parameters: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()


def main():
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)
    logging.info(f"Connected to database at '{DB_PATH}'.")

    try:
        # Step 1: Create scaling_params table
        create_scaling_params_table(conn, SCALING_PARAMS_TABLE)

        # Step 2: Rename 'open_price' to 'actual_open_price'
        rename_column(conn, SRC_TABLE, "open_price", "actual_open_price")

        # Step 3: Add new columns 'open_price' and 'symbol_numeric_normalized'
        new_columns = {
            "open_price": "REAL",
            "symbol_numeric_normalized": "REAL"
        }
        add_new_columns(conn, SRC_TABLE, new_columns)

        # Step 4: Load 'features' table data into DataFrame
        df = load_features_table_into_df(conn, SRC_TABLE)

        # Step 5: Perform standard scaling on selected columns
        #    Only normalize 'actual_open_price' (to 'open_price') and 'symbol_numeric' (to 'symbol_numeric_normalized')
        columns_to_normalize = NORMALIZE_COLUMNS.copy()
        # Remove any columns that are in EXCLUDED_COLUMNS
        columns_to_normalize = {orig: new for orig, new in columns_to_normalize.items() if orig not in EXCLUDED_COLUMNS}

        logging.info(f"Columns to normalize: {columns_to_normalize}")

        df_scaled, scaling_params = standard_scale_selected_columns(df, columns_to_normalize)

        # Log scaling_params for verification
        logging.info("Scaling Parameters:")
        logging.info("\n" + tabulate(scaling_params, headers='keys', tablefmt='psql'))

        # Step 6: Update the 'features' table with normalized values
        update_columns = list(columns_to_normalize.values())
        update_features_table(conn, df_scaled, update_columns, BATCH_SIZE)

        # Step 7: Insert scaling parameters into 'scaling_params' table
        insert_scaling_params(conn, scaling_params, SCALING_PARAMS_TABLE)

        logging.info("Normalization process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during normalization: {e}")
    finally:
        conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
