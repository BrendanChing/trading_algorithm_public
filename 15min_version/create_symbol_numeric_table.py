import os
import sqlite3
import psycopg2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Step 1: Connect to SQLite and retrieve data ---

# Path to your SQLite database file
SQLITE_DB_PATH = 'stock_data.db'  # adjust if needed

try:
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    sqlite_cursor = sqlite_conn.cursor()
    logger.info("Connected to SQLite database.")

    # Query the features table for distinct rows
    sqlite_cursor.execute("""
        SELECT DISTINCT symbol, symbol_numeric, symbol_numeric_normalized 
        FROM features
    """)
    rows = sqlite_cursor.fetchall()
    logger.info(f"Fetched {len(rows)} rows from SQLite 'features' table.")
except Exception as e:
    logger.error(f"Error connecting to SQLite: {e}")
    raise
finally:
    sqlite_conn.close()


# --- Step 2: Connect to PostgreSQL and create/update the table ---

# Get PostgreSQL connection string from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set!")

try:
    pg_conn = psycopg2.connect(DATABASE_URL)
    pg_cursor = pg_conn.cursor()
    logger.info("Connected to PostgreSQL database.")

    # Create a new table for the features data (if it doesn't already exist)
    # Here we name it 'features_pg'; adjust the table name as needed.
    pg_cursor.execute("""
        CREATE TABLE IF NOT EXISTS features_pg (
            symbol TEXT PRIMARY KEY,
            symbol_numeric INTEGER,
            symbol_numeric_normalized REAL
        )
    """)
    pg_conn.commit()
    logger.info("Created table 'features_pg' (if not existing).")

    # Insert or update rows from SQLite into PostgreSQL.
    # Using an UPSERT so that if the symbol already exists, we update its values.
    for row in rows:
        symbol, symbol_numeric, symbol_numeric_normalized = row
        pg_cursor.execute("""
            INSERT INTO features_pg (symbol, symbol_numeric, symbol_numeric_normalized)
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE
            SET symbol_numeric = EXCLUDED.symbol_numeric,
                symbol_numeric_normalized = EXCLUDED.symbol_numeric_normalized
        """, (symbol, symbol_numeric, symbol_numeric_normalized))
    
    pg_conn.commit()
    logger.info("Inserted/updated all rows into 'features_pg'.")
except Exception as e:
    logger.error(f"Error working with PostgreSQL: {e}")
    raise
finally:
    if pg_cursor:
        pg_cursor.close()
    if pg_conn:
        pg_conn.close()
