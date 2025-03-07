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
SRC_TABLE = "features"      # Source table is now 'features'
DEST_TABLE = "features"     # Destination is the same table
BATCH_SIZE = 10000          # Number of rows to update per batch commit

# Your symbol list
SYMBOLS = [
    'SEDG', 'FL', 'AAP', 'FIVN', 'INTC', 'BTU', 'GH', 'JWN', 'URBN', 'ADNT', 'CNK',
    'AA', 'NTGR', 'CVGW', 'CAL', 'RVLV', 'FVRR', 'BOX', 'NTLA', 'LUV', 'SMTC',
    'RAMP', 'VRNT', 'XRAY', 'HP', 'BEN', 'PDCO', 'NEP', 'BBIO', 'SSTK', 'MOS',
    'JD', 'PINS', 'AMN', 'DBX', 'RMBS', 'FUN', 'QFIN', 'CEVA', 'CPRI', 'TNDM',
    'MRO', 'CNP', 'DAR', 'ASB', 'SPT', 'VIRT', 'VNO', 'RF', 'AROC', 'EQT', 'BEAM',
    'DOC', 'BWA', 'HAL', 'CAG', 'BLFS', 'FTI', 'TPR', 'PLAB', 'IRDM', 'KURA',
    'NTCT', 'TDC', 'PLYM', 'NI', 'IPG', 'PRGO', 'RPD', 'SLM', 'ICHR', 'KELYA',
    'CARG', 'UCTT', 'CRSP', 'MYGN', 'BRX', 'DVN', 'LNC', 'DAVA', 'FCPT', 'RRC'
]

# Approximate steps (in units of 15-min bars).
STEP_SIZE_MAP = {
    '15min': 1,
    '30min': 2,
    '1hour': 4,
    '1day': 26,    # ~6.5 hours trading day => 26 bars
    '1month': 520  # ~20 trading days => 20*26
}

# Define new features
NEW_FEATURES = [
    'dist_min_ma_26',
    'dist_max_ma_26',
    'dist_min_ma_260',
    'dist_max_ma_260',
    'dist_ma_step1',
    'dist_ma_step26',
    'std_dev_open_price_16',
    'rsi_14'
]

PRICE_COL = 'open_price'

def get_table_schema(conn, table_name):
    """
    Returns a list of (col_name, col_type) from the given table, based on PRAGMA table_info.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    cur.close()

    schema = [(r[1], r[2]) for r in rows]
    return schema

def add_new_feature_columns(conn, table, new_features):
    """
    Adds new feature columns to the table if they do not already exist.
    All new features are of type REAL.
    """
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        existing_columns = set(row[1] for row in cur.fetchall())

        needed_cols = [feat for feat in new_features if feat not in existing_columns]

        for col in needed_cols:
            logging.info(f"Adding column '{col}' to table '{table}'...")
            alter_sql = f"ALTER TABLE {table} ADD COLUMN {col} REAL"
            cur.execute(alter_sql)

        conn.commit()
        cur.close()

        if needed_cols:
            logging.info(f"Added columns: {', '.join(needed_cols)}")
        else:
            logging.info("No new columns needed to be added.")

    except sqlite3.Error as e:
        logging.error(f"Failed to add new feature columns: {e}")
        conn.rollback()
        raise

def create_indexes(conn):
    """
    Creates indexes on 'symbol', 'date', and 'time' columns in 'features' table
    to optimize query performance.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_symbol_date_time
            ON features (symbol, date, time)
        """)
        logging.info("Created index on 'features' table for (symbol, date, time).")
        conn.commit()
        cursor.close()
    except sqlite3.Error as e:
        logging.error(f"Failed to create indexes: {e}")
        conn.rollback()
        raise

def load_features_table(conn, symbols):
    """
    Loads data from the 'features' table for the specified symbols into a DataFrame.
    """
    logging.info("Loading data from 'features' table...")
    query = f"""
        SELECT *
        FROM features
        WHERE symbol IN ({','.join(['?']*len(symbols))})
        ORDER BY symbol, date, time
    """
    df = pd.read_sql_query(query, conn, params=symbols)
    logging.info(f"Loaded {len(df)} rows from 'features' table.")
    return df

def compute_new_features(df):
    """
    Computes the new features and adds them to the DataFrame.
    """
    logging.info("Computing new features...")

    # Initialize new feature columns with NaN
    for feat in NEW_FEATURES:
        df[feat] = np.nan

    # Group by symbol to compute features per symbol
    grouped = df.groupby('symbol')

    for symbol, group in grouped:
        logging.info(f"Processing symbol: {symbol}")
        df.loc[group.index, NEW_FEATURES] = compute_features_per_symbol(group).values

    return df

def compute_features_per_symbol(group):
    """
    Computes the new features for a single symbol's DataFrame.
    Returns a DataFrame with new feature columns.
    """
    symbol_df = group.sort_values(['date', 'time']).copy()

    # Compute 5-point moving average
    symbol_df['ma_5'] = symbol_df[PRICE_COL].rolling(window=5, min_periods=1).mean()

    # Distance to min/max of moving average over 26 and 260 points
    for window in [26, 260]:
        ma_min = symbol_df['ma_5'].rolling(window=window, min_periods=1).min()
        ma_max = symbol_df['ma_5'].rolling(window=window, min_periods=1).max()
        symbol_df[f'dist_min_ma_{window}'] = symbol_df['ma_5'] - ma_min
        symbol_df[f'dist_max_ma_{window}'] = symbol_df['ma_5'] - ma_max

    # Distance to moving average with step sizes
    # Step size 1 (15min)
    ma_step1 = symbol_df['ma_5'].shift(1)  # shift by 1 step
    symbol_df['dist_ma_step1'] = symbol_df[PRICE_COL] - ma_step1

    # Step size 26 (1 day)
    ma_step26 = symbol_df['ma_5'].shift(26)  # shift by 26 steps
    symbol_df['dist_ma_step26'] = symbol_df[PRICE_COL] - ma_step26

    # Standard deviation of open_price over 16 points
    symbol_df['std_dev_open_price_16'] = symbol_df[PRICE_COL].rolling(window=16, min_periods=1).std()

    # Relative Strength Index (RSI) over 14 periods
    symbol_df['rsi_14'] = compute_rsi(symbol_df[PRICE_COL], window=14)

    # Select the new features
    new_features_df = symbol_df[NEW_FEATURES].copy()

    return new_features_df

def compute_rsi(series, window=14):
    """
    Computes the Relative Strength Index (RSI) for a given price series.
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle division by zero
    rsi = rsi.fillna(0)

    return rsi

def update_features_table(conn, df, new_features, batch_size=BATCH_SIZE):
    """
    Updates the 'features' table with the new features in batches.
    """
    logging.info("Updating 'features' table with new features...")

    cols = ['symbol', 'date', 'time'] + new_features
    update_cols = new_features
    placeholders = ", ".join(["?"] * len(update_cols))
    update_sql = f"""
        UPDATE features
        SET {', '.join([f"{col}=?" for col in update_cols])}
        WHERE symbol = ? AND date = ? AND time = ?
    """

    cur = conn.cursor()

    total_rows = len(df)
    logging.info(f"Total rows to update: {total_rows}")

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch_df = df.iloc[start:end]

        records = []
        for _, row in batch_df.iterrows():
            values = [row[col] if not pd.isna(row[col]) else None for col in update_cols]
            key = [row['symbol'], row['date'], row['time']]
            records.append(values + key)

        try:
            cur.executemany(update_sql, records)
            conn.commit()
            logging.info(f"Updated rows {start} to {end - 1}")
        except sqlite3.Error as e:
            logging.error(f"Failed to update batch {start} to {end - 1}: {e}")
            conn.rollback()

    cur.close()
    logging.info("Finished updating 'features' table with new features.")

def verify_updates(conn, sample_size=5):
    """
    Verifies that the new feature columns have been correctly populated
    by displaying a sample of updated records.
    """
    try:
        cursor = conn.cursor()
        # Fetch a sample of records where new features are updated
        cursor.execute(f"""
            SELECT symbol, date, time, {', '.join(NEW_FEATURES)}
            FROM features
            WHERE {' OR '.join([f"{feat} IS NOT NULL" for feat in NEW_FEATURES])}
            LIMIT ?
        """, (sample_size,))

        records = cursor.fetchall()
        if records:
            # Define column headers
            headers = ['Symbol', 'Date', 'Time'] + NEW_FEATURES
            # Display the sample in a table format
            table = tabulate(records, headers=headers, tablefmt='psql')
            logging.info("Sample of updated records with new features:")
            logging.info("\n" + table)
        else:
            logging.warning("No records were updated with new features. Please check the feature calculations.")

    except sqlite3.Error as e:
        logging.error(f"Failed to verify updates: {e}")
        raise

def main():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DB_PATH)
        logging.info(f"Connected to database at '{DB_PATH}'.")

        # Step 1: Add new feature columns to 'features' table
        add_new_feature_columns(conn, DEST_TABLE, NEW_FEATURES)

        # Step 2: Create indexes to optimize performance
        create_indexes(conn)

        # Step 3: Load data from 'features' table
        df = load_features_table(conn, SYMBOLS)
        if df.empty:
            logging.warning("No data found in 'features' table. Exiting.")
            conn.close()
            return

        # Step 4: Compute new features
        df = compute_new_features(df)

        # Step 5: Update 'features' table with new features
        update_features_table(conn, df, NEW_FEATURES, BATCH_SIZE)

        # Step 6: Verify the updates with a sample of records
        verify_updates(conn, sample_size=5)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
