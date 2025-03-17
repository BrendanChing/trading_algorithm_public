import os
import sqlite3
import logging
from dotenv import load_dotenv

# Load environment variables from a .env file if available
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Step 1: Connect to SQLite and retrieve data from the 'features' table ---

# Path to your SQLite database file
SQLITE_DB_PATH = 'stock_data.db'  # Adjust if needed

try:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    logger.info("Connected to SQLite database.")
    
    # Query the features table for distinct rows with valid numeric values
    cursor.execute("""
        SELECT DISTINCT symbol, symbol_numeric, symbol_numeric_normalized 
        FROM features
        WHERE symbol_numeric IS NOT NULL AND symbol_numeric_normalized IS NOT NULL
    """)
    rows = cursor.fetchall()
    logger.info(f"Fetched {len(rows)} rows from SQLite 'features' table.")
    
    # Debug: Print the first 5 rows to inspect data
    logger.info(f"First 5 rows: {rows[:5]}")
except Exception as e:
    logger.error(f"Error connecting to SQLite: {e}")
    raise
finally:
    conn.close()

# --- Step 2: Create a new table in SQLite and insert/update the data ---

try:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Create a new table for the symbol numeric values (if it doesn't already exist)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symbol_numeric_sqlite (
            symbol TEXT PRIMARY KEY,
            symbol_numeric INTEGER,
            symbol_numeric_normalized REAL
        )
    """)
    conn.commit()
    logger.info("Created table 'symbol_numeric_sqlite' (if not existing).")
    
    # Insert or update rows from the features table using SQLite UPSERT syntax
    for row in rows:
        symbol, symbol_numeric, symbol_numeric_normalized = row
        cursor.execute("""
            INSERT INTO symbol_numeric_sqlite (symbol, symbol_numeric, symbol_numeric_normalized)
            VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET 
                symbol_numeric = excluded.symbol_numeric,
                symbol_numeric_normalized = excluded.symbol_numeric_normalized
        """, (symbol, symbol_numeric, symbol_numeric_normalized))
    
    conn.commit()
    logger.info("Inserted/updated all rows into 'symbol_numeric_sqlite'.")
except Exception as e:
    logger.error(f"Error working with SQLite for new table: {e}")
    raise
finally:
    if cursor is not None:
        cursor.close()
    if conn is not None:
        conn.close()
