import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =====================================================
# Database path
db_path = 'stock_data.db'  # Update with your actual database path

# =====================================================
def column_exists(connection, table_name, column_name):
    """
    Check if a column exists in a given table.

    Parameters:
        connection (sqlite3.Connection): SQLite database connection.
        table_name (str): Name of the table.
        column_name (str): Name of the column to check.

    Returns:
        bool: True if column exists, False otherwise.
    """
    cursor = connection.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for col in columns:
        if col[1] == column_name:
            return True
    return False

# =====================================================
def add_symbol_numeric_column(connection, table_name):
    """
    Add a new column 'symbol_numeric' to the specified table.

    Parameters:
        connection (sqlite3.Connection): SQLite database connection.
        table_name (str): Name of the table to alter.

    Returns:
        None
    """
    cursor = connection.cursor()
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN symbol_numeric INTEGER")
        connection.commit()
        logging.info(f"Added 'symbol_numeric' column to '{table_name}' table.")
    except sqlite3.OperationalError as e:
        logging.error(f"Failed to add 'symbol_numeric' column to '{table_name}' table: {e}")

# =====================================================
def fetch_symbol_mapping(connection):
    """
    Fetch mapping of symbols to symbol_numeric from the 'features' table.

    Parameters:
        connection (sqlite3.Connection): SQLite database connection.

    Returns:
        dict: Dictionary mapping symbol (str) to symbol_numeric (int).
    """
    cursor = connection.cursor()
    query = """
        SELECT DISTINCT symbol, symbol_numeric
        FROM features
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    mapping = {row[0]: row[1] for row in rows}
    logging.info(f"Fetched {len(mapping)} symbol mappings from 'features' table.")
    return mapping

# =====================================================
def update_historical_15min(connection, symbol_mapping):
    """
    Update the 'symbol_numeric' column in 'historical_15min' table based on symbol.

    Parameters:
        connection (sqlite3.Connection): SQLite database connection.
        symbol_mapping (dict): Dictionary mapping symbol (str) to symbol_numeric (int).

    Returns:
        int: Number of rows updated.
    """
    cursor = connection.cursor()
    rows_updated = 0
    try:
        # Start a transaction
        cursor.execute("BEGIN TRANSACTION;")
        
        for symbol, symbol_num in symbol_mapping.items():
            # Update all rows where symbol matches and symbol_numeric is NULL
            cursor.execute("""
                UPDATE historical_15min
                SET symbol_numeric = ?
                WHERE symbol = ? AND symbol_numeric IS NULL
            """, (symbol_num, symbol))
            updated = cursor.rowcount
            if updated > 0:
                logging.debug(f"Updated {updated} rows for symbol '{symbol}' with symbol_numeric={symbol_num}.")
                rows_updated += updated
        
        # Commit the transaction
        connection.commit()
        logging.info(f"Total rows updated in 'historical_15min': {rows_updated}")
    except sqlite3.Error as e:
        connection.rollback()
        logging.error(f"Failed to update 'symbol_numeric' in 'historical_15min' table: {e}")
    
    return rows_updated

# =====================================================
def main():
    # Connect to the SQLite database
    try:
        connection = sqlite3.connect(db_path)
        logging.info(f"Connected to database at '{db_path}'.")
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to database: {e}")
        return

    table_name = 'historical_15min'

    # Check if 'symbol_numeric' column exists
    if column_exists(connection, table_name, 'symbol_numeric'):
        logging.info(f"'symbol_numeric' column already exists in '{table_name}' table.")
    else:
        # Add 'symbol_numeric' column
        add_symbol_numeric_column(connection, table_name)
    
    # Fetch symbol to symbol_numeric mapping from 'features' table
    symbol_mapping = fetch_symbol_mapping(connection)
    
    if not symbol_mapping:
        logging.error("No symbol mappings found. Exiting.")
        connection.close()
        return
    
    # Update 'historical_15min' table with 'symbol_numeric'
    rows_updated = update_historical_15min(connection, symbol_mapping)
    
    # Optionally, you can fetch and display some statistics
    cursor = connection.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]
    
    cursor.execute(f"SELECT COUNT(symbol_numeric) FROM {table_name}")
    symbol_numeric_filled = cursor.fetchone()[0]
    
    logging.info(f"'historical_15min' Table Stats:")
    logging.info(f"Total Rows: {total_rows}")
    logging.info(f"Rows with 'symbol_numeric' filled: {symbol_numeric_filled}")
    logging.info(f"Rows remaining with 'symbol_numeric' NULL: {total_rows - symbol_numeric_filled}")
    
    # Close the connection
    connection.close()
    logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
