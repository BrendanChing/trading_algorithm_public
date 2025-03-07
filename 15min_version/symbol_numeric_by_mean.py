import sqlite3
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Database path
DB_PATH = 'stock_data.db'
TABLE_NAME = 'historical_15min'

def update_symbol_numeric(conn, table=TABLE_NAME):
    """
    Updates the 'symbol_numeric' column by ordering symbols based on their mean 'open_price'.
    Assigns integers starting from 0 up to 81.
    """
    try:
        # Fetch symbols and their mean open_price
        query = f"""
            SELECT symbol, AVG(open_price) as mean_open_price
            FROM {table}
            GROUP BY symbol
            ORDER BY mean_open_price ASC
        """
        df = pd.read_sql_query(query, conn)
        
        logging.info(f"Fetched mean open_price for {len(df)} symbols.")
        
        # Assign numeric values starting from 0
        df['symbol_numeric'] = range(len(df))
        
        # Create a dictionary mapping symbol to symbol_numeric
        symbol_numeric_map = pd.Series(df.symbol_numeric.values, index=df.symbol).to_dict()
        
        logging.info("Created symbol to numeric mapping.")
        
        # Begin transaction
        cursor = conn.cursor()
        conn.execute('BEGIN TRANSACTION;')
        
        # Update each symbol with its numeric value
        for symbol, numeric in symbol_numeric_map.items():
            cursor.execute(f"""
                UPDATE {table}
                SET symbol_numeric = ?
                WHERE symbol = ?
            """, (numeric, symbol))
        
        # Commit transaction
        conn.commit()
        logging.info("Updated 'symbol_numeric' for all symbols successfully.")
        
    except Exception as e:
        logging.error(f"Failed to update 'symbol_numeric': {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def main():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DB_PATH)
        logging.info(f"Connected to database at '{DB_PATH}'.")
        
        # Update symbol_numeric
        update_symbol_numeric(conn, TABLE_NAME)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
