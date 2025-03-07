import sqlite3
import pandas as pd
import numpy as np
import logging
from tabulate import tabulate

DB_PATH = "stock_data.db"
TABLE_NAME = "features"

# The symbols you used before
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

############################
# 1) Ensure REWARD column exists
############################

def add_reward_column_if_missing(conn, table=TABLE_NAME, col_name="reward"):
    """
    Checks if 'reward' column exists in the 'features' table. If not, adds it as REAL.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    existing_cols = set(row[1] for row in cur.fetchall())

    if col_name not in existing_cols:
        logging.info(f"Adding column '{col_name}' to table '{table}'...")
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} REAL")
        conn.commit()
        logging.info(f"Column '{col_name}' added successfully.")
    else:
        logging.info(f"Column '{col_name}' already exists in table '{table}'.")
    cur.close()

############################
# 2) Create Indexes for Performance
############################

def create_indexes(connection):
    """
    Creates indexes on 'symbol', 'date', and 'time' columns in 'features' and 'historical_15min' tables
    to optimize join performance.
    """
    try:
        cursor = connection.cursor()
        # Create index on 'features' table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_symbol_date_time
            ON features (symbol, date, time)
        """)
        logging.info("Created index on 'features' table for (symbol, date, time).")
        
        # Create index on 'historical_15min' table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_symbol_date_time
            ON historical_15min (symbol, date, time)
        """)
        logging.info("Created index on 'historical_15min' table for (symbol, date, time).")
        
        connection.commit()
    except sqlite3.Error as e:
        logging.error(f"Failed to create indexes: {e}")
        connection.rollback()
        raise

############################
# 3) Load data & compute reward
############################

def load_data_for_symbol(symbol):
    """
    Loads all rows for the given symbol from 'features' into a DataFrame,
    including the rowid so we can update them. Returns a sorted DataFrame.
    Uses 'actual_open_price' instead of 'open_price'.
    """
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT 
            rowid as db_rowid,
            symbol,
            date,
            time,
            actual_open_price,
            high_price,
            low_price,
            close_price
        FROM {TABLE_NAME}
        WHERE symbol = '{symbol}'
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Create a datetime column for sorting
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], utc=True)
    df.sort_values(by='datetime', inplace=True, ignore_index=True)
    return df


def compute_reward_labels(df, lookback=16, lookforward=16, up_mult=2.5, down_mult=2.5):
    """
    Assigns reward=1 if the 'up threshold' is reached before the 'down threshold'
    within the next 'lookforward' bars. Otherwise reward=0.

    - up_threshold = current_actual_open + up_mult * rolling_std
    - down_threshold = current_actual_open - down_mult * rolling_std
    - rolling_std is computed over the last 'lookback' bars of actual_open_price.

    If neither threshold is hit, reward=0.
    If we can't look forward (last 'lookforward' bars), reward=NaN by default.
    """
    # 1) Compute rolling std of actual_open_price
    df['std_open'] = df['actual_open_price'].rolling(window=lookback, min_periods=lookback).std()

    # 2) Initialize reward to NaN
    df['reward'] = np.nan

    # We'll iterate up to len(df) - lookforward, because we need that many bars ahead
    last_valid_index = len(df) - lookforward

    for i in range(lookback, last_valid_index):
        current_std = df.loc[i, 'std_open']
        if pd.isna(current_std):
            # If we don't have a valid std, set reward=0
            df.at[i, 'reward'] = 0
            continue

        current_actual_open = df.loc[i, 'actual_open_price']

        up_threshold = current_actual_open + up_mult * current_std
        down_threshold = current_actual_open - down_mult * current_std

        # The future slice to check
        future_slice = df.iloc[i+1 : i+1+lookforward]

        # We scan bar by bar in chronological order
        # If we hit down_threshold first => reward=0
        # If we hit up_threshold first => reward=1
        # If neither is reached => reward=0

        reward = 0  # default if we never hit either threshold
        for idx_fut, row_fut in future_slice.iterrows():
            if row_fut['low_price'] <= down_threshold:
                reward = 0
                break
            elif row_fut['high_price'] >= up_threshold:
                reward = 1
                break

        df.at[i, 'reward'] = reward

    # 3) If we want the last 'lookforward' rows to remain NaN
    df.loc[last_valid_index:, 'reward'] = np.nan

    # Clean up if you like
    df.drop(columns='std_open', inplace=True)

    return df

############################
# 4) Update reward in DB with Batch Updates
############################

def update_reward_in_db(df, batch_size=10000):
    """
    Updates the 'reward' column in the same table using rowid as the match.
    We do it in batches for speed.
    Uses 'actual_open_price' instead of 'open_price'.
    """
    if 'reward' not in df.columns:
        return

    # Connect once here
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    num_rows = len(df)
    logging.info(f"Updating reward for {num_rows} rows...")

    # Build the update statement
    sql = f"""
    UPDATE {TABLE_NAME}
    SET reward = ?
    WHERE rowid = ?
    """

    start_idx = 0
    while start_idx < num_rows:
        end_idx = min(start_idx + batch_size, num_rows)
        batch_df = df.iloc[start_idx:end_idx]

        param_list = []
        for _, row in batch_df.iterrows():
            # Convert NaN -> None
            val = row['reward']
            if pd.isna(val):
                val = None
            rowid = row['db_rowid']
            param_list.append((val, rowid))

        cur.executemany(sql, param_list)
        conn.commit()

        logging.info(f"Updated reward for rows [{start_idx}..{end_idx-1}]")
        start_idx = end_idx

    cur.close()
    conn.close()

############################
# 5) Verify updates
############################

def verify_updates(connection, sample_size=5):
    """
    Verifies that the 'reward' column has been correctly populated
    by displaying a sample of updated records.
    """
    try:
        cursor = connection.cursor()
        # Fetch a sample of records where 'reward' is not NULL
        cursor.execute(f"""
            SELECT symbol, date, time, actual_open_price, reward
            FROM {TABLE_NAME}
            WHERE reward IS NOT NULL
            LIMIT ?
        """, (sample_size,))

        records = cursor.fetchall()
        if records:
            # Define column headers
            headers = ['Symbol', 'Date', 'Time', 'Actual Open Price', 'Reward']
            # Display the sample in a table format
            table = tabulate(records, headers=headers, tablefmt='psql')
            logging.info("Sample of updated records:")
            logging.info("\n" + table)
        else:
            logging.warning("No records were updated. Please check the alignment criteria.")
    except sqlite3.Error as e:
        logging.error(f"Failed to verify updates: {e}")
        raise

############################
# 6) Main
############################

def main():
    # 1) Ensure 'reward' column exists
    conn = sqlite3.connect(DB_PATH)
    add_reward_column_if_missing(conn, TABLE_NAME, col_name="reward")
    create_indexes(conn)
    conn.close()

    for sym in SYMBOLS:
        logging.info(f"Loading data for {sym}...")
        df = load_data_for_symbol(sym)
        if df.empty:
            logging.info(f"No rows found for {sym}. Skipping.")
            continue

        logging.info(f"Computing reward labels for {sym} (rows={len(df)})...")
        df = compute_reward_labels(df, lookback=16, lookforward=16)

        # 2) Update the DB with the 'reward' column
        update_reward_in_db(df, batch_size=10000)

    # 3) Verify updates for all symbols
    # Reconnect to the database
    conn = sqlite3.connect(DB_PATH)
    verify_updates(conn, sample_size=5)
    conn.close()

if __name__ == "__main__":
    main()
