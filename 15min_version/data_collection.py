import requests
import sqlite3
import logging
import time
from datetime import datetime, timedelta
import pytz

# 1) Configuration
FMP_API_KEY = "J7E8IP0IRthaJebTxv2MODaeT1uFB6nP"  # <-- Replace with your actual API key
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

DB_PATH = "stock_data.db"

# Date Range (inclusive)
START_DATE_STR = "2022-01-01"
END_DATE_STR   = "2025-01-13"

# Convert to datetime (UTC)
START_DATE = datetime.strptime(START_DATE_STR, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
END_DATE   = datetime.strptime(END_DATE_STR, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

# Symbol list (from your earlier snippet)
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

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_historical_15min_table():
    """
    Drops the 'historical_15min' table if it exists, then creates it.
    WARNING: This will delete all existing data in 'historical_15min'.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop if exists
    cur.execute("DROP TABLE IF EXISTS historical_15min")

    # Re-create table
    cur.execute("""
        CREATE TABLE historical_15min (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            open_price REAL NOT NULL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume REAL
        )
    """)
    conn.commit()
    cur.close()
    conn.close()



def fetch_15min_data_monthly(symbol, start_date, end_date, api_key):
    """
    Fetch intraday 15-minute data month-by-month from start_date to end_date,
    using the 'from' and 'to' query parameters. 
    Returns a list of dicts with keys: 'date', 'open', 'high', 'low', 'close', 'volume'.
    """
    monthly_data = []
    current_date = start_date

    while current_date <= end_date:
        # Move forward ~1 month
        next_month = current_date + timedelta(days=30)
        # Do not exceed end_date
        chunk_end = min(next_month, end_date)

        formatted_start = current_date.strftime('%Y-%m-%d')
        formatted_end   = chunk_end.strftime('%Y-%m-%d')

        # Build URL
        api_url = (
            f"{FMP_BASE_URL}/historical-chart/15min/{symbol}"
            f"?from={formatted_start}&to={formatted_end}&apikey={api_key}"
        )
        logging.info(f"Requesting 15-min data for {symbol} from {formatted_start} to {formatted_end}...")

        resp = requests.get(api_url)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                # Data might be in descending order, so let's reverse
                # to get oldest -> newest
                data.reverse()
                monthly_data.extend(data)
            else:
                logging.warning(f"Unexpected format: {data}")
        else:
            logging.warning(f"Failed to fetch 15-min data for {symbol} "
                            f"[{formatted_start} -> {formatted_end}]: HTTP {resp.status_code}")

        # Move to the next chunk
        current_date = next_month

        # Sleep to avoid rate limits
        time.sleep(0.25)

    return monthly_data


def insert_15min_data(symbol, bars):
    """
    Inserts each bar's data into 'historical_15min', ignoring any bar
    missing required fields. (We assume bars is in ascending time order.)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for bar in bars:
        # Example bar: {
        #   "date": "2023-01-25 15:00:00",
        #   "open": 123.4,
        #   "high": 125.0,
        #   "low": 122.7,
        #   "close": 124.1,
        #   "volume": 3500000
        # }
        bar_date_str = bar.get("date")
        if not bar_date_str:
            continue

        # date in form "YYYY-MM-DD HH:MM:SS"
        parts = bar_date_str.split()
        if len(parts) != 2:
            continue
        date_str, time_str = parts

        open_price  = bar.get("open")
        high_price  = bar.get("high")
        low_price   = bar.get("low")
        close_price = bar.get("close")
        volume      = bar.get("volume")

        # Basic validation
        if open_price is None or high_price is None or low_price is None or close_price is None:
            continue

        # Insert row
        cur.execute("""
            INSERT INTO historical_15min (symbol, date, time, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, date_str, time_str, open_price, high_price, low_price, close_price, volume))

    conn.commit()
    cur.close()
    conn.close()


def main():
    ensure_historical_15min_table()

    for symbol in SYMBOLS:
        logging.info(f"==== Collecting 15-min data for {symbol} ====")
        bars = fetch_15min_data_monthly(symbol, START_DATE, END_DATE, FMP_API_KEY)
        if not bars:
            logging.info(f"No data returned for {symbol}.")
            continue
        
        logging.info(f"Fetched {len(bars)} bars for {symbol}. Now inserting into DB...")
        insert_15min_data(symbol, bars)
        logging.info(f"Inserted {len(bars)} bars for {symbol}.\n")


if __name__ == "__main__":
    main()
