import sqlite3
import logging

DB_PATH = "stock_data.db"
FEATURES_TABLE = "features"
NUM_ROWS_TO_DROP = 1600

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1) Get all distinct symbols
    cur.execute(f"SELECT DISTINCT symbol FROM {FEATURES_TABLE}")
    symbols = [row[0] for row in cur.fetchall()]

    logging.info(f"Found {len(symbols)} symbols in '{FEATURES_TABLE}'.")

    # 2) For each symbol, delete the first 1600 rows by ascending (date, time)
    for sym in symbols:
        logging.info(f"Dropping first {NUM_ROWS_TO_DROP} rows for symbol={sym}...")

        delete_sql = f"""
            DELETE FROM {FEATURES_TABLE}
            WHERE rowid IN (
                SELECT rowid
                FROM {FEATURES_TABLE}
                WHERE symbol = ?
                ORDER BY date, time
                LIMIT {NUM_ROWS_TO_DROP}
            )
        """
        cur.execute(delete_sql, (sym,))
        deleted_count = cur.rowcount  # number of rows actually deleted
        logging.info(f"Deleted {deleted_count} rows for symbol={sym}.")

    conn.commit()
    cur.close()
    conn.close()
    logging.info("Finished dropping rows.")

if __name__ == "__main__":
    main()
