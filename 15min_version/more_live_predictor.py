import sqlite3
import requests
import pandas as pd
import logging
import math
import time
import pytz
from datetime import datetime, timedelta
import schedule
import threading
import traceback
import xgboost as xgb
import time as time_module
from dateutil.relativedelta import relativedelta
import exchange_calendars as ec
from hidden_functionality import calculate_features
from feature_mapping import REAL_TO_PUBLIC
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("live_predictor_updated.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration Variables
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
LIVE_TABLE = "live_15min"
PREDICTIONS_TABLE = "predictions_15min"
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
MODEL_PATH = "xgboost_15min_more.json"

# Database Path
DB_PATH = "stock_data.db"

excluded_symbols = []

# NEW: Global set for symbols that must be skipped if data is incomplete
GLOBAL_SYMBOLS_TO_SKIP = set()

############################
# 1) Setup Data Function
############################

def get_symbol_numeric_normalized(conn, symbol):
    """
    Retrieves the numeric code and its normalized version for a given stock symbol 
    from the 'features' table.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT symbol_numeric, symbol_numeric_normalized 
            FROM features 
            WHERE symbol = ? 
            LIMIT 1
        """, (symbol,))
        result = cursor.fetchone()
        if result:
            return result  # (symbol_numeric, symbol_numeric_normalized)
        else:
            raise ValueError(f"Symbol {symbol} not found in the database.")
    except Exception as e:
        logger.error(f"Error retrieving numeric codes for symbol {symbol}: {e}")
        raise

def setup_data():
    """
    Creates live_15min table if it does not exists with the last 2000 15min datapoints.
    Fetches data in batches of 200, adjusting for non-trading days. Scheduled daily at 8:30 AM ET.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        logger.info("Connected to the SQLite database for setup_data().")

        # Create new
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {LIVE_TABLE} (
                rowid INTEGER PRIMARY KEY,
                symbol TEXT,
                date TEXT,
                time TEXT,
                actual_open_price REAL,        -- Raw price from API
                open_price REAL,               -- Normalized price (model expects 'open_price')
                symbol_numeric INTEGER,
                symbol_numeric_normalized REAL,
                ATR_16 REAL,
                dist_min_ma_26 REAL,
                dist_max_ma_26 REAL,
                dist_min_ma_260 REAL,
                dist_max_ma_260 REAL,
                dist_ma_step1 REAL,
                dist_ma_step26 REAL,
                std_dev_open_price_16 REAL,
                rsi_14 REAL,
                dist_min_ma_26_z REAL,
                dist_max_ma_26_z REAL,
                dist_min_ma_260_z REAL,
                dist_max_ma_260_z REAL,
                dist_ma_step1_z REAL,
                dist_ma_step26_z REAL,
                std_dev_open_price_16_z REAL,
                rsi_14_z REAL,
                UNIQUE(symbol, date, time)
            )
        """)
        logger.info(f"Created table '{LIVE_TABLE}' with UNIQUE constraint on (symbol, date, time).")

        # Define batch parameters
        batch_size = 200
        total_datapoints = 2000
        data_points_per_day = 26
        trading_days_per_batch = math.ceil(batch_size / data_points_per_day)
        calendar_days_per_batch = trading_days_per_batch + 6
        batches_required = math.ceil(total_datapoints / batch_size)

        for symbol in SYMBOLS:
            try:
                # Retrieve numeric mapping from the features table
                symbol_codes = get_symbol_numeric_normalized(conn, symbol)
                symbol_numeric, symbol_numeric_normalized = symbol_codes
            except ValueError as ve:
                logger.error(f"Symbol '{symbol}': {ve}")
                excluded_symbols.append(symbol)
                continue

            all_data = []
            end_time = datetime.now(pytz.timezone('US/Eastern'))
            current_end_time = end_time
            logger.info(f"Starting data fetch for symbol: {symbol}")

            for batch_num in range(batches_required):
                # Calculate start time for the current batch
                current_start_time = current_end_time - timedelta(days=calendar_days_per_batch)
                formatted_start = current_start_time.strftime('%Y-%m-%d')
                formatted_end = current_end_time.strftime('%Y-%m-%d')

                url = f"{FMP_BASE_URL}/historical-chart/15min/{symbol}"
                params = {
                    'from': formatted_start,
                    'to': formatted_end,
                    'apikey': FMP_API_KEY
                }

                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    logger.debug(f"Batch {batch_num+1} response for {symbol}: {data}")
                except requests.exceptions.HTTPError as http_err:
                    logger.error(f"HTTP error for {symbol}: {http_err}")
                    excluded_symbols.append(symbol)
                    break
                except Exception as err:
                    logger.error(f"Error fetching {symbol}: {err}")
                    excluded_symbols.append(symbol)
                    break

                if isinstance(data, list):
                    fetched_points = len(data)
                    all_data.extend(data)
                    logger.info(f"Fetched {fetched_points} datapoints for {symbol} (batch {batch_num+1}).")

                    if fetched_points < batch_size:
                        logger.info(f"Less than batch_size for {symbol}, ending fetch.")
                        break
                else:
                    logger.warning(f"Unexpected data format for {symbol}: {data}")
                    excluded_symbols.append(symbol)
                    break

                current_end_time = current_start_time - timedelta(minutes=15)
                time_module.sleep(1)

            if not all_data:
                logger.warning(f"No data for {symbol}, skipping.")
                excluded_symbols.append(symbol)
                continue

            # Sort ascending (oldest first)
            df = pd.DataFrame(all_data).sort_values('date')
            if len(df) > total_datapoints:
                df = df.tail(total_datapoints)
                logger.info(f"Trimmed data for {symbol} to last {total_datapoints} rows.")

            if df.empty:
                excluded_symbols.append(symbol)
                continue

            if 'date' not in df.columns:
                logger.error(f"No 'date' column for {symbol}, skipping.")
                excluded_symbols.append(symbol)
                continue

            # Convert 'date' to datetime objects
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if df['date'].isnull().any():
                logger.error(f"Invalid date data for {symbol}, skipping.")
                excluded_symbols.append(symbol)
                continue

            # Extract date and time
            df['date_only'] = df['date'].dt.strftime('%Y-%m-%d')
            df['time'] = df['date'].dt.strftime('%H:%M')

            # We expect these columns from FMP historical API
            required_columns = ['date_only', 'time', 'open']
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                logger.error(f"Missing {missing_cols} for {symbol}, skipping.")
                excluded_symbols.append(symbol)
                continue

            # Rename 'open' -> 'actual_open_price'
            df = df[required_columns]
            df.rename(columns={'date_only': 'date', 'open': 'actual_open_price'}, inplace=True)

            # Basic columns
            df['symbol'] = symbol
            df['symbol_numeric'] = symbol_numeric
            df['symbol_numeric_normalized'] = symbol_numeric_normalized

            # We also have a 'open_price' column in the DB for normalized price,
            # so initialize it to None; it will be filled by calculate_features()
            df['open_price'] = None
            
            # Initialize ATR_16 (to be filled later)
            df['ATR_16'] = None

            # Initialize the various new feature columns
            new_features = [
                'dist_min_ma_26', 'dist_max_ma_26', 'dist_min_ma_260', 'dist_max_ma_260',
                'dist_ma_step1', 'dist_ma_step26', 'std_dev_open_price_16', 'rsi_14',
                'dist_min_ma_26_z', 'dist_max_ma_26_z', 'dist_min_ma_260_z', 'dist_max_ma_260_z',
                'dist_ma_step1_z', 'dist_ma_step26_z', 'std_dev_open_price_16_z', 'rsi_14_z'
            ]
            for feature in new_features:
                df[feature] = None

            # Deduplicate (symbol, date, time)
            before_dedup = len(df)
            df.drop_duplicates(subset=['symbol', 'date', 'time'], keep='last', inplace=True)
            after_dedup = len(df)
            if (before_dedup - after_dedup) > 0:
                logger.warning(f"Removed {before_dedup - after_dedup} duplicates for {symbol} in setup_data().")

            # Insert into the live_15min table
            try:
                df.to_sql(LIVE_TABLE, conn, if_exists='append', index=False)
                logger.info(f"Inserted {len(df)} records for {symbol} into {LIVE_TABLE}.")
            except sqlite3.IntegrityError as ie:
                logger.error(f"Integrity error for {symbol} in setup_data(): {ie}")
                excluded_symbols.append(symbol)
                continue

        # Commit changes
        conn.commit()
        logger.info(f"Setup data completed. Excluded symbols: {excluded_symbols}")

    except Exception as e:
        logger.error(f"setup_data failed: {e}")
        logger.debug(traceback.format_exc())
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed in setup_data.")


############################
# 2) Checking/Filling Missing Data (NEW)
############################

def check_and_fill_missing_data():
    """
    Checks the live_15min table for missing 15-minute intervals over the last 3 months,
    but ONLY for actual trading times on NYSE from 9:30 to 15:45 ET (skips weekends/holidays).
    
    If a missing bar is within the last 100 intervals from 'now_rounded', attempt to fetch 
    it from the FMP historical 15min API. Otherwise, fill by midpoint interpolation.

    Returns:
        set: symbols_to_skip containing symbols that couldn't fill recent missing bars.
    """
    logger.info("check_and_fill_missing_data() called!")
    symbols_to_skip = set()
    conn = None
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1) Determine 'now_rounded' in Eastern
        eastern = pytz.timezone('US/Eastern')
        current_time_et = datetime.now(eastern)
        rounded_minute = (current_time_et.minute // 15) * 15
        now_rounded = current_time_et.replace(minute=rounded_minute, second=0, microsecond=0)

        # 2) We look back 3 months from now_rounded
        earliest_dt_et = now_rounded - relativedelta(months=3)

        # 3) Build the NYSE calendar and fetch the schedule
        nyse = ec.get_calendar("XNYS")  # Official NYSE schedule
        # Get the schedule from earliest to now (in date terms)
        schedule = nyse.schedule(
            start_date=earliest_dt_et.date(), 
            end_date=now_rounded.date()
        )
        # => schedule has columns 'market_open' & 'market_close' (both in UTC)

        # 4) Adjust the close from 16:00 to 15:45 for each trading day
        #    so we effectively end 15 minutes earlier
        schedule['market_close'] = schedule['market_close'] - pd.Timedelta(minutes=15)

        # 5) Generate all trading minutes in UTC for that adjusted schedule
        #    This will give 1-minute bars from 9:30..15:45
        all_minutes_utc = nyse.minutes_for_period(schedule)

        # 6) Convert them to Eastern time
        all_minutes_et = all_minutes_utc.tz_convert("US/Eastern")

        # 7) Keep only those that are on a 15-minute boundary, i.e. minute % 15 == 0
        def is_15min_bar(ts):
            # e.g. 09:30 => minute=30, which is 30 % 15 ==0 => keep
            # e.g. 09:31 => minute=31 => skip
            return (ts.minute % 15 == 0) and (ts.second == 0)

        mask_15min = [is_15min_bar(ts) for ts in all_minutes_et]
        all_15min_et = all_minutes_et[mask_15min]

        # Filter out any bars after 'now_rounded' in ET (just in case)
        all_15min_et = all_15min_et[all_15min_et <= now_rounded]

        # Convert each bar to a naive datetime for string comparison
        should_exist = []
        for dt_ts in all_15min_et:
            dt_naive = dt_ts.replace(tzinfo=None)
            should_exist.append(dt_naive)

        def parse_existing_bar(date_str, time_str):
            """Parses the existing row's date/time into a datetime."""
            full_str = f"{date_str} {time_str}"
            try:
                return datetime.strptime(full_str, "%Y-%m-%d %H:%M")
            except:
                return None

        # We'll define the time window in strings for the DB query
        start_str = earliest_dt_et.strftime('%Y-%m-%d %H:%M')
        end_str   = now_rounded.strftime('%Y-%m-%d %H:%M')

        for symbol in SYMBOLS:
            # 8) Gather existing rows for this symbol in that time window
            rows_q = f"""
                SELECT date, time
                FROM {LIVE_TABLE}
                WHERE symbol = ?
                  AND (date || ' ' || time) >= ?
                  AND (date || ' ' || time) <= ?
                ORDER BY date, time
            """
            rows = cursor.execute(rows_q, (symbol, start_str, end_str)).fetchall()
            
            existing_dt = set()
            for (d, t) in rows:
                dt_obj = parse_existing_bar(d, t)
                if dt_obj:
                    existing_dt.add(dt_obj)

            missing_dt = [dt for dt in should_exist if dt not in existing_dt]
            if not missing_dt:
                logger.info(f"No missing bars for {symbol} in the last 3 months (9:30..15:45).")
                continue
            
            logger.warning(f"Found {len(missing_dt)} missing bars for {symbol}. Attempting to fill...")

            # 9) Insert/fill function
            def insert_missing_bar(sym, dt_target, use_midpoint=False):
                """
                Insert or fill the missing bar at dt_target for sym.
                If use_midpoint=False, fetch from FMP. If that fails => return False.
                If use_midpoint=True, fill open price by midpoint of nearest known bars.
                Return True if success, False otherwise.
                """
                dt_date_str = dt_target.strftime('%Y-%m-%d')
                dt_time_str = dt_target.strftime('%H:%M')

                if use_midpoint:
                    dt_full_str = f"{dt_date_str} {dt_time_str}"
                    prev_q = f"""
                        SELECT actual_open_price 
                        FROM {LIVE_TABLE}
                        WHERE symbol = ?
                          AND (date || ' ' || time) < ?
                        ORDER BY date DESC, time DESC
                        LIMIT 1
                    """
                    next_q = f"""
                        SELECT actual_open_price
                        FROM {LIVE_TABLE}
                        WHERE symbol = ?
                          AND (date || ' ' || time) > ?
                        ORDER BY date ASC, time ASC
                        LIMIT 1
                    """
                    prev_price = cursor.execute(prev_q, (sym, dt_full_str)).fetchone()
                    next_price = cursor.execute(next_q, (sym, dt_full_str)).fetchone()
                    if not prev_price or not next_price:
                        return False
                    p1, p2 = prev_price[0], next_price[0]
                    if p1 is None or p2 is None:
                        return False
                    midpoint = (p1 + p2) / 2.0
                    try:
                        sql = f"""
                            INSERT INTO {LIVE_TABLE}
                            (symbol, date, time, actual_open_price)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(symbol, date, time) DO NOTHING
                        """
                        cursor.execute(sql, (sym, dt_date_str, dt_time_str, midpoint))
                        return True
                    except Exception as e:
                        logger.error(f"Error inserting midpoint for {sym} {dt_full_str}: {e}")
                        return False
                else:
                    # Attempt to fetch from FMP
                    from_date = (dt_target - timedelta(days=1)).strftime('%Y-%m-%d')
                    to_date   = (dt_target + timedelta(days=1)).strftime('%Y-%m-%d')
                    try:
                        url = f"{FMP_BASE_URL}/historical-chart/15min/{sym}"
                        params = {'apikey': FMP_API_KEY, 'from': from_date, 'to': to_date}
                        resp = requests.get(url, params=params)
                        resp.raise_for_status()
                        data = resp.json()
                        found_bar = None
                        for bar in data:
                            bar_dt_str = bar.get('date')
                            if not bar_dt_str:
                                continue
                            try:
                                bar_dt = datetime.strptime(bar_dt_str, "%Y-%m-%d %H:%M:%S")
                            except:
                                continue
                            if bar_dt == dt_target:
                                found_bar = bar
                                break
                        if not found_bar:
                            return False
                        actual_open = found_bar.get('open')
                        if actual_open is None:
                            return False
                        insert_sql = f"""
                            INSERT INTO {LIVE_TABLE}
                            (symbol, date, time, actual_open_price)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(symbol, date, time) DO NOTHING
                        """
                        cursor.execute(insert_sql, (sym, dt_date_str, dt_time_str, actual_open))
                        return True
                    except Exception as ex:
                        logger.error(f"Error fetching bar for {sym} at {dt_target}: {ex}")
                        return False

            # 10) If missing bar is within last 100 intervals => attempt FMP fetch, else midpoint
            # each interval is 15min => 100 intervals => 100*15=1500min => ~25 hours
            now_minus_100bars = now_rounded - timedelta(minutes=100*15)

            for dt_missing_ in missing_dt:
                if dt_missing_ >= now_minus_100bars:
                    # Attempt direct fetch
                    ok = insert_missing_bar(symbol, dt_missing_, use_midpoint=False)
                    if not ok:
                        logger.error(f"Cannot fill bar for {symbol} within last 100 intervals => skip symbol.")
                        symbols_to_skip.add(symbol)
                        break
                else:
                    # older => midpoint
                    ok = insert_missing_bar(symbol, dt_missing_, use_midpoint=True)
                    if not ok:
                        logger.warning(f"Midpoint fill failed for {symbol} at {dt_missing_}. Continuing...")

        conn.commit()
        logger.info(f"Completed check_and_fill_missing_data() with exchange_calendars (9:30..15:45). symbols_to_skip: {symbols_to_skip}")
        return symbols_to_skip

    except Exception as e:
        logger.error(f"check_and_fill_missing_data() failed: {e}")
        traceback_str = traceback.format_exc()
        logger.debug(traceback_str)
        return set()
    finally:
        if conn:
            conn.close()
            logger.info("DB connection closed in check_and_fill_missing_data.")


def run_check_and_fill_missing():
    """
    Wrapper that merges the returned skip set into the GLOBAL_SYMBOLS_TO_SKIP.
    """
    new_skips = check_and_fill_missing_data()
    if new_skips:
        GLOBAL_SYMBOLS_TO_SKIP.update(new_skips)
    logger.info(f"GLOBAL_SYMBOLS_TO_SKIP updated: {GLOBAL_SYMBOLS_TO_SKIP}")

############################
# 3) Live Data Ingestion
############################

def fetch_and_insert_live_data():
    """
    Every 15 minutes, fetch the real-time askPrice for each symbol 
    and upsert into live_15min if symbol not in GLOBAL_SYMBOLS_TO_SKIP.

    In this version, we explicitly SELECT to check row existence,
    then do an INSERT or UPDATE accordingly (instead of ON CONFLICT).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        logger.debug("DB connection established in fetch_and_insert_live_data().")
        
        # 1) Round to 15-minute boundary
        eastern = pytz.timezone('US/Eastern')
        current_time_et = datetime.now(eastern)
        rounded_minute = (current_time_et.minute // 15) * 15
        rounded_time = current_time_et.replace(
            minute=rounded_minute, second=0, microsecond=0
        )
        date_str = rounded_time.strftime('%Y-%m-%d')
        time_str = rounded_time.strftime('%H:%M:%S')

        logger.debug(
            f"fetch_and_insert_live_data() => date={date_str}, time={time_str}"
        )

        for symbol in SYMBOLS:
            # Skip symbols with incomplete data
            if symbol in GLOBAL_SYMBOLS_TO_SKIP:
                logger.warning(
                    f"Skipping {symbol} in fetch_and_insert_live_data() due to incomplete data."
                )
                continue

            # 2) Get symbol_numeric
            try:
                symbol_codes = get_symbol_numeric_normalized(conn, symbol)
                symbol_numeric, symbol_numeric_normalized = symbol_codes
            except ValueError as ve:
                logger.error(f"Missing numeric codes for '{symbol}': {ve}")
                continue

            # 3) Fetch real-time from FMP
            url = f"{FMP_BASE_URL}/stock/full/real-time-price/{symbol}"
            params = {'apikey': FMP_API_KEY}
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"API response for {symbol}: {data}")
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error for {symbol}: {http_err}")
                continue
            except Exception as err:
                logger.error(f"Other error for {symbol}: {err}")
                continue

            # 4) Extract askPrice
            if isinstance(data, list) and len(data) > 0:
                ask_price = data[0].get('askPrice')
            elif isinstance(data, dict):
                ask_price = data.get('askPrice')
            else:
                ask_price = None

            # If ask_price is None or zero, skip
            if not ask_price or ask_price <= 0:
                logger.warning(f"No valid askPrice for {symbol} at {date_str} {time_str}, skipping.")
                continue

            logger.debug(f"askPrice for {symbol}: {ask_price}")

            # 5) Check if row already exists
            cursor.execute(
                f"SELECT 1 FROM {LIVE_TABLE} WHERE symbol=? AND date=? AND time=?",
                (symbol, date_str, time_str)
            )
            row_exists = cursor.fetchone()

            # 6) If row exists => UPDATE; else => INSERT
            if row_exists:
                # We do an UPDATE
                logger.info(f"Row for {symbol} at {date_str} {time_str} already exists. Updating actual_open_price.")
                try:
                    cursor.execute(f"""
                        UPDATE {LIVE_TABLE}
                        SET actual_open_price = ?,
                            symbol_numeric = ?,
                            symbol_numeric_normalized = ?,
                            open_price = NULL
                        WHERE symbol=? AND date=? AND time=?
                    """, (
                        ask_price,
                        symbol_numeric,
                        symbol_numeric_normalized,
                        symbol,
                        date_str,
                        time_str
                    ))
                    logger.info(f"Updated row for {symbol} at {date_str} {time_str} with askPrice={ask_price}.")
                except Exception as e:
                    logger.error(f"Failed to update live data for {symbol}: {e}")
                    logger.debug(traceback.format_exc())
                    continue
            else:
                # We do an INSERT
                logger.info(f"Inserting new row for {symbol} at {date_str} {time_str}.")
                try:
                    cursor.execute(f"""
                        INSERT INTO {LIVE_TABLE}
                        (symbol, date, time, actual_open_price, open_price, symbol_numeric, symbol_numeric_normalized)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        date_str,
                        time_str,
                        ask_price,
                        None,
                        symbol_numeric,
                        symbol_numeric_normalized
                    ))
                    logger.info(f"Inserted live data for {symbol} at {date_str} {time_str} with askPrice={ask_price}.")
                except Exception as e:
                    logger.error(f"Failed to insert live data for {symbol}: {e}")
                    logger.debug(traceback.format_exc())
                    continue

        conn.commit()
        cursor.close()
        conn.close()
        logger.debug("DB connection closed in fetch_and_insert_live_data().")

    except Exception as e:
        logger.error(f"fetch_and_insert_live_data failed: {e}")
        logger.debug(traceback.format_exc())
        if 'conn' in locals():
            conn.close()

############################
# 5) Prediction
############################

def load_trained_model():
    try:
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        logger.info("XGBoost Booster model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.debug(traceback.format_exc())
        return None
    
def rename_features(df):
    for real_name, public_name in REAL_TO_PUBLIC.items():
        if real_name in df.columns:
            df.rename(columns={real_name: public_name}, inplace=True)
    return df    

def make_predictions(model):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
            
        cursor.execute(f"DROP TABLE IF EXISTS {PREDICTIONS_TABLE}")
        logger.info(f"Dropped table '{PREDICTIONS_TABLE}' if it existed.")
        cursor.execute(f"""
            CREATE TABLE {PREDICTIONS_TABLE} (
                symbol TEXT,
                date TEXT,
                time TEXT,
                open_price REAL,
                prediction REAL,
                stop_loss REAL,
                take_profit REAL
            )
        """)
        logger.info(f"Created table '{PREDICTIONS_TABLE}'.")
        conn.commit()
        
        df = pd.read_sql_query(f"SELECT * FROM {LIVE_TABLE} ORDER BY date DESC, time DESC", conn)
        if df.empty:
            logger.warning("No data in live_15min for predictions.")
            return
        
        # skip symbols in GLOBAL_SYMBOLS_TO_SKIP
        df = df[~df['symbol'].isin(GLOBAL_SYMBOLS_TO_SKIP)]
        if df.empty:
            logger.warning("All symbols are skipped due to missing data.")
            return
        
        df_latest = df.sort_values(['symbol','date','time']).groupby('symbol').tail(1)

        feature_cols = [
            'Feature_A','Feature_B','Feature_C','Feature_D',
            'Feature_E','Feature_F','Feature_G','Feature_H',
            'Feature_I','Feature_J','Feature_K','Feature_L',
            'Feature_M','Feature_N','Feature_O','Feature_P',
            'Feature_Q','Feature_R','Feature_S',
            'open_price','symbol_numeric_normalized',
            'dist_min_ma_26_z','dist_max_ma_26_z','dist_min_ma_260_z','dist_max_ma_260_z',
            'dist_ma_step1_z','dist_ma_step26_z','std_dev_open_price_16_z','rsi_14_z'
        ]

        missing_feats = [c for c in feature_cols if c not in df_latest.columns]
        if missing_feats:
            logger.error(f"Missing features for prediction: {missing_feats}")
            return

        X = df_latest[feature_cols].fillna(0)
        dmatrix = xgb.DMatrix(X, feature_names=X.columns.tolist())
        predictions = model.predict(dmatrix)

        df_latest['prediction'] = predictions
        df_latest['stop_loss'] = -3 * df_latest['ATR_16']
        df_latest['take_profit'] = 4 * df_latest['ATR_16']

        predictions_df = df_latest[['symbol','date','time','actual_open_price','prediction','stop_loss','take_profit']].copy()
        predictions_df.rename(columns={'actual_open_price':'open_price'}, inplace=True)

        predictions_df.to_sql(PREDICTIONS_TABLE, conn, if_exists='append', index=False)
        logger.info(f"Inserted {len(predictions_df)} prediction records into {PREDICTIONS_TABLE}.")

        # Optionally print top predictions
        df_sorted = predictions_df.sort_values('prediction', ascending=False)
        top_predictions = df_sorted[['symbol','prediction']].reset_index(drop=True)
        print("Top Predictions:")
        print(top_predictions.to_string(index=False))

        logger.info("Predictions made and stored successfully.")
    except Exception as e:
        logger.error(f"make_predictions failed: {e}")
        logger.debug(traceback.format_exc())
    finally:
        conn.close()

############################
# 6) Main Scheduling
############################

def run_setup_data():
    setup_data()

def run_live_prediction():
    df = calculate_features()
    df = rename_features(df)
    logger.info("run_live_prediction() called!")
    fetch_and_insert_live_data()
    calculate_features()
    model = load_trained_model()
    if model:
        make_predictions(model)

def scheduler():
    """
    Schedules:
    - setup_data at 8:30 AM ET
    - check_and_fill_missing_data() 5 min before each bar (9:25..15:40 ET)
    - run_live_prediction at each bar (9:30..15:45 ET)
    """
    try:
        eastern = pytz.timezone('US/Eastern')
        local_tz = pytz.timezone('Europe/London')  # If server is in the UK

        def generate_local_trading_times(start_time_str, end_time_str, interval_minutes=15):
            today = datetime.now(eastern).date()
            start_et = eastern.localize(datetime.combine(today, datetime.strptime(start_time_str, "%H:%M").time()))
            end_et   = eastern.localize(datetime.combine(today, datetime.strptime(end_time_str, "%H:%M").time()))
            times = []
            current_et = start_et
            while current_et <= end_et:
                local_dt = current_et.astimezone(local_tz)
                times.append(local_dt.strftime("%H:%M"))
                current_et += timedelta(minutes=interval_minutes)
            return times

        # schedule setup_data daily at 8:30 AM ET =>    local
        setup_time_et = "9:00"
        setup_dt_et = eastern.localize(datetime.strptime(setup_time_et, "%H:%M"))
        setup_time_local = setup_dt_et.astimezone(local_tz).strftime("%H:%M")
        schedule.every().day.at(setup_time_local).do(run_setup_data)
        logger.info(f"Scheduled setup_data daily at {setup_time_local} local (= {setup_time_et} ET).")

        # generate times for 9:30..15:45 ET
        main_trading_times_local = generate_local_trading_times("09:30","15:45",15)
        for bar_local_str in main_trading_times_local:
            # parse local time, subtract 5 min
            dt_today_local = datetime.now(local_tz).date()
            bar_time_obj = datetime.strptime(bar_local_str, "%H:%M").time()
            bar_dt_local = local_tz.localize(datetime.combine(dt_today_local, bar_time_obj))
            pre_check_dt = bar_dt_local - timedelta(minutes=5)
            pre_check_str = pre_check_dt.strftime("%H:%M")

            # schedule pre-check
            schedule.every().day.at(pre_check_str).do(run_check_and_fill_missing)
            # schedule main
            schedule.every().day.at(bar_local_str).do(run_live_prediction)

            logger.info(f"Scheduled pre-check at {pre_check_str} local, then live_prediction at {bar_local_str} local.")

        logger.info("Scheduler setup complete. Entering loop.")

        while True:
            schedule.run_pending()
            time_module.sleep(1)
    except Exception as e:
        logger.error(f"Scheduler encountered error: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    # Run initial tasks
    # setup_data()              # use only if want to run immediately upon running script--generally wait for scheduling
    # run_live_prediction()     # use only if want to run immediately upon running script--generally wait for scheduling 
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Live predictor stopped by user.")
