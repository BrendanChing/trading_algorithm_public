import sqlite3
import pandas as pd
import logging
import traceback

DB_PATH = "stock_data.db"
TABLE_NAME = "simulation_15min"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def add_std_dev_16_column():
    """
    Adds the std_dev_16 column to simulation_15min if it doesn't exist yet.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
        columns = {row[1] for row in cursor.fetchall()}
        if "std_dev_16" not in columns:
            cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN std_dev_16 REAL")
            conn.commit()
            logger.info(f"Added column 'std_dev_16' to {TABLE_NAME}.")
        else:
            logger.info(f"Column 'std_dev_16' already exists in {TABLE_NAME}.")

    except Exception as e:
        logger.error(f"Failed to add column 'std_dev_16': {e}")
        logger.debug(traceback.format_exc())
    finally:
        if conn:
            conn.close()

def compute_and_update_std_dev_16():
    """
    Loads all rows from simulation_15min (with rowid, symbol, date, time, actual_open_price),
    computes the 16-bar rolling std dev for each symbol in chronological order,
    and updates the std_dev_16 column.
    
    If insufficient data (<16 bars) for a row, std_dev_16 is set to NULL.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)

        # 1) Load entire table in chronological order by (symbol, date, time).
        #    We'll require rowid, symbol, date, time, actual_open_price
        sql = f"""
            SELECT rowid, symbol, date, time, actual_open_price
            FROM {TABLE_NAME}
            ORDER BY symbol, date, time
        """
        df = pd.read_sql_query(sql, conn)
        logger.info(f"Loaded {len(df)} rows from {TABLE_NAME}.")

        if df.empty:
            logger.warning("No data to process for std_dev_16.")
            return

        # 2) We'll compute rolling std for each symbol over actual_open_price
        #    min_periods=16 means we only get a std dev if we have 16 values.
        #    We'll preserve the row order with a rolling approach.
        def rolling_std_16(series):
            return series.rolling(window=16, min_periods=16).std()

        # Group by symbol, then transform
        df["std_dev_16"] = df.groupby("symbol")["actual_open_price"].transform(rolling_std_16)

        # 3) Now we have a DataFrame with possibly NaN in rows that can't compute std dev
        #    We'll convert those to None for the DB update.
        df["std_dev_16"] = df["std_dev_16"].where(df["std_dev_16"].notna(), None)

        # 4) Prepare updates: (std_dev_value, rowid) for each row
        #    We use a list of tuples, then executemany
        updates = []
        for index, row in df.iterrows():
            row_id = row["rowid"]
            std_val = row["std_dev_16"]
            # might be float or None
            updates.append((std_val, row_id))

        if not updates:
            logger.warning("No rows to update for std_dev_16.")
            return

        cursor = conn.cursor()
        # We'll do batch update
        cursor.executemany(
            f"UPDATE {TABLE_NAME} SET std_dev_16 = ? WHERE rowid = ?",
            updates
        )
        conn.commit()
        logger.info(f"Updated std_dev_16 for {len(updates)} rows in {TABLE_NAME}.")

    except Exception as e:
        logger.error(f"Error computing/updating std_dev_16: {e}")
        logger.debug(traceback.format_exc())
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    add_std_dev_16_column()
    compute_and_update_std_dev_16()
