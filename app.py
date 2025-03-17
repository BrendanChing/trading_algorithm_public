from flask import Flask, jsonify, render_template, request, redirect, flash, session, send_from_directory
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import json
import logging
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('APP_SECRET_KEY')  # Required for flash messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the Supabase PostgreSQL connection string from environment variables.
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set!")

# Global variables for table names
LIVE_TABLE = "live_15min"
PREDICTIONS_TABLE = "predictions_15min"

# Assume these globals are defined elsewhere in your code:
# SYMBOLS: list of stock symbols.
# GLOBAL_SYMBOLS_TO_SKIP: a set of symbols to skip.
# (Also, any helper functions like get_symbol_numeric_normalized have been refactored to use PostgreSQL.)
# For example, if you have a refactored get_symbol_numeric_normalized that queries table "symbol_numeric":
def get_symbol_numeric_normalized(conn, symbol):
    """
    Retrieves the numeric code and its normalized version for a given stock symbol 
    from the 'symbol_numeric' table.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT symbol_numeric, symbol_numeric_normalized 
            FROM symbol_numeric
            WHERE symbol = %s
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

# ----------------------------
# Routes using PostgreSQL below
# ----------------------------

@app.route('/subscribe', methods=['POST'])
def subscribe():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    subscription = request.json.get('subscription')
    if not subscription:
        return jsonify({"error": "Subscription data missing"}), 400

    user_id = session['user_id']
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                subscription TEXT NOT NULL
            )
        """)
        cursor.execute("INSERT INTO subscriptions (user_id, subscription) VALUES (%s, %s)", (user_id, json.dumps(subscription)))
        conn.commit()
    except Exception as e:
        logger.error(f"Error in subscribe: {e}")
        return jsonify({"error": "Error saving subscription"}), 500
    finally:
        conn.close()
    return jsonify({"message": "Subscription saved successfully"}), 200

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT id, password, role FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
        except Exception as e:
            logger.error(f"Error in login: {e}")
            user = None
        finally:
            conn.close()
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['role'] = user[2]
            flash("Login successful!", "success")
            return redirect('/')
        else:
            flash("Invalid credentials!", "error")
    return render_template('login.html')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("You need to be logged in to access this page.", "error")
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/update-portfolio', methods=['POST'])
@login_required
def update_portfolio():
    try:
        portfolio_value = float(request.form.get('portfolio_value'))
        user_id = session['user_id']
        logger.info(f"Received portfolio value: {portfolio_value} for user_id: {user_id}")

        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_portfolio (
                user_id INTEGER PRIMARY KEY,
                portfolio_value REAL
            )
        """)
        cursor.execute("""
            INSERT INTO user_portfolio (user_id, portfolio_value)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE SET portfolio_value = EXCLUDED.portfolio_value
        """, (user_id, portfolio_value))
        conn.commit()
    except Exception as e:
        logger.error(f"Error updating portfolio value: {e}")
        flash("Failed to update portfolio value.", "error")
    finally:
        conn.close()
    flash("Portfolio value updated successfully!", "success")
    return redirect('/full_table')

@app.route('/admin')
@login_required
def admin_dashboard():
    if session.get('role') != 'admin':
        flash("Access denied! Admins only.", "error")
        return redirect('/')
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role FROM users")
        users = cursor.fetchall()
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        users = []
    finally:
        conn.close()
    return render_template('admin_dashboard.html', users=users)

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect('/')

@app.route('/')
@login_required
def home():
    user_id = session['user_id']
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, prediction, date, time FROM predictions 
            WHERE prediction >= 0.82 OR prediction <= 0.3 
            ORDER BY prediction DESC
        """)
        results = cursor.fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching predictions in home: {e}")
        results = []
    buy = [row for row in results if row[1] >= 0.7]
    sell = [row for row in results if row[1] <= 0.3]

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM bought_stock WHERE user_id = %s ORDER BY timestamp DESC", (user_id,))
        symbols = cursor.fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching user trades: {e}")
        symbols = []

    trades = []
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        for symbol_tuple in symbols:
            symbol = symbol_tuple[0]
            cursor.execute("SELECT prediction, stop_loss FROM predictions WHERE symbol = %s", (symbol,))
            prediction_data = cursor.fetchone()
            if prediction_data:
                prediction, stop_loss = prediction_data
                trades.append({"symbol": symbol, "prediction": prediction, "stop_loss": stop_loss})
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching predictions for trades: {e}")
        trades = []
    return render_template('index.html', buy=buy, sell=sell, trades=trades)

@app.route('/add_trade', methods=['POST'])
@login_required
def add_trade():
    symbol = request.form.get('symbol').upper()
    user_id = session['user_id']
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE symbol = %s", (symbol,))
        exists = cursor.fetchone()[0]
        conn.close()
    except Exception as e:
        logger.error(f"Error checking predictions for add_trade: {e}")
        exists = 0
    if not exists:
        flash(f"Symbol '{symbol}' not found in predictions!", "error")
        return redirect('/')
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO bought_stock (symbol, user_id) VALUES (%s, %s)", (symbol, user_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error inserting trade for {symbol}: {e}")
        flash("Error adding trade.", "error")
        return redirect('/')
    return redirect('/')

@app.route('/remove_trade', methods=['POST'])
@login_required
def remove_trade():
    symbol = request.form.get('symbol').upper()
    user_id = session['user_id']
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bought_stock WHERE symbol = %s AND user_id = %s", (symbol, user_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error removing trade for {symbol}: {e}")
        flash("Error removing trade.", "error")
        return redirect('/')
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM bought_stock WHERE user_id = %s", (user_id,))
        remaining_symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching remaining trades for user: {e}")
        remaining_symbols = []
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        if remaining_symbols:
            placeholders = ', '.join(['%s'] * len(remaining_symbols))
            query = f"SELECT symbol, prediction, stop_loss FROM predictions WHERE symbol IN ({placeholders})"
            cursor.execute(query, tuple(remaining_symbols))
            updated_trades = cursor.fetchall()
        else:
            updated_trades = []
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching updated predictions: {e}")
        updated_trades = []
    flash(f"Successfully removed '{symbol}' from your trades!", "success")
    return render_template('index.html', trades=updated_trades)

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/full_table')
@login_required
def full_table():
    return render_template('full_table.html')

@app.route('/predictions_15min')
@login_required
def predictions_15min():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, date, time, open_price, prediction, stop_loss, take_profit FROM predictions_15min")
        rows = cursor.fetchall()
        predictions = []
        for row in rows:
            predictions.append({
                "symbol": row[0],
                "date": row[1],
                "time": row[2],
                "open_price": row[3],
                "prediction": row[4],
                "stop_loss": row[5],
                "take_profit": row[6]
            })
        conn.close()
        return jsonify(predictions)
    except Exception as e:
        app.logger.error(f"Error fetching predictions: {e}")
        return jsonify({"error": "Error fetching predictions"}), 500

@app.route('/full_table_15min')
@login_required
def full_table_15min():
    return render_template('full_table_15min.html')

@app.route('/predictions', methods=['GET'])
@login_required
def get_predictions():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, date, time, open_price, prediction, stop_loss, take_profit, capital_invested
            FROM predictions
            ORDER BY prediction DESC
        """)
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return jsonify({"error": "Error fetching predictions"}), 500

    predictions = [
        {
            "symbol": row[0],
            "date": row[1],
            "time": row[2],
            "open_price": row[3],
            "prediction": row[4],
            "stop_loss": row[5],
            "take_profit": row[6],
            "capital_invested": row[7]
        }
        for row in rows
    ]
    return jsonify(predictions)

if __name__ == '__main__':
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                subscription TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bought_stock (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        logger.error(f"Error creating initial tables: {e}")
    finally:
        conn.close()
    app.run(debug=True)
