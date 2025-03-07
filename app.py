from flask import Flask, jsonify, render_template, request, redirect, flash, session
from functools import wraps
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from pywebpush import webpush, WebPushException
import json
import logging
from flask import send_from_directory
import psycopg2
import os

DATABASE_URL = os.getenv('DATABASE_URL')

conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cur = conn.cursor()

cur.execute("SELECT NOW()")
print(cur.fetchone())

cur.close()
conn.close()

app = Flask(__name__)
app.secret_key = os.getenv('APP_SECRET_KEY')  # Required for flash messages

logging.basicConfig(level=logging.INFO)

VAPID_PRIVATE_KEY = os.getenv('VAPID_PRIVATE_KEY')

VAPID_PUBLIC_KEY = 'BBS10xMUSLRw3g9PhIGFner4uQbPYfcSTQ8vF3RMSa6JO6DDJ4fwgYr1k6AtqAkyYPMxB7F9CikPHINnHaPix8c'

app.config['VAPID_PRIVATE_KEY'] = VAPID_PRIVATE_KEY
app.config['VAPID_PUBLIC_KEY'] = VAPID_PUBLIC_KEY

@app.route('/service-worker.js')
def service_worker():
    return send_from_directory('.', 'service-worker.js', mimetype='application/javascript')

@app.route('/vapid_public_key', methods=['GET'])
def vapid_public_key():
    return jsonify({"publicKey": app.config['VAPID_PUBLIC_KEY']})

@app.route('/test_notification', methods=['POST'])
def test_notification():
    logging.info("Received request data: %s", request.data)
    message = request.json.get('message', 'Test notification!')
    connection = sqlite3.connect('user_trades.db')
    cursor = connection.cursor()
    cursor.execute("SELECT subscription FROM subscriptions")
    subscriptions = cursor.fetchall()
    connection.close()

    for subscription in subscriptions:
        subscription_info = json.loads(subscription[0])
        send_push_notification(subscription_info, message)

    return jsonify({"message": "Test notification sent!"}), 200


def send_push_notification(subscription_info, message):
    try:
        webpush(
            subscription_info=subscription_info,
            data=json.dumps({"message": message}),
            vapid_private_key=app.config['VAPID_PRIVATE_KEY'],
            vapid_claims={
                "sub": "mailto:your-email@example.com"
            }
        )
    except WebPushException as e:
        logging.error(f"WebPushException: {e}")


@app.route('/subscribe', methods=['POST'])
def subscribe():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    subscription = request.json.get('subscription')
    if not subscription:
        return jsonify({"error": "Subscription data missing"}), 400

    user_id = session['user_id']
    connection = sqlite3.connect('user_trades.db')
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subscription TEXT NOT NULL
        )
    """)
    cursor.execute("INSERT INTO subscriptions (user_id, subscription) VALUES (?, ?)", (user_id, json.dumps(subscription)))
    connection.commit()
    connection.close()

    return jsonify({"message": "Subscription saved successfully"}), 200

@app.route('/send_notification', methods=['POST'])
def send_notification():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 401

    message = request.json.get('message')
    if not message:
        return jsonify({"error": "Message is required"}), 400

    connection = sqlite3.connect('user_trades.db')
    cursor = connection.cursor()
    cursor.execute("SELECT subscription FROM subscriptions")
    subscriptions = cursor.fetchall()
    connection.close()

    for subscription in subscriptions:
        subscription_info = json.loads(subscription[0])
        send_push_notification(subscription_info, message)

    return jsonify({"message": "Notifications sent successfully"}), 200

# Route for login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        connection = sqlite3.connect('user_trades.db')
        cursor = connection.cursor()
        cursor.execute("SELECT id, password, role FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        connection.close()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['role'] = user[2]
            flash("Login successful!", "success")
            return redirect('/')
        else:
            flash("Invalid credentials!", "error")

    return render_template('login.html')


# Login required decorator
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
    """
    Updates the user's portfolio value in the database.
    """
    try:
        portfolio_value = float(request.form.get('portfolio_value'))
        user_id = session['user_id']  # Get the logged-in user's ID

        logging.info(f"Received portfolio value: {portfolio_value} for user_id: {user_id}")

        # Save the portfolio value to the database
        connection = sqlite3.connect('user_trades.db')
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO user_portfolio (user_id, portfolio_value)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET portfolio_value = excluded.portfolio_value
        """, (user_id, portfolio_value))
        connection.commit()
        connection.close()

        flash("Portfolio value updated successfully!", "success")
    except Exception as e:
        logging.error(f"Error updating portfolio value: {e}")
        flash("Failed to update portfolio value.", "error")

    return redirect('/full_table')


@app.route('/admin')
@login_required
def admin_dashboard():
    if session.get('role') != 'admin':
        flash("Access denied! Admins only.", "error")
        return redirect('/')

    connection = sqlite3.connect('user_trades.db')
    cursor = connection.cursor()
    cursor.execute("SELECT id, username, role FROM users")
    users = cursor.fetchall()
    connection.close()

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

    # Fetch buy and sell predictions
    connection = sqlite3.connect('stock_data.db')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT symbol, prediction, date, time FROM predictions 
        WHERE prediction >= 0.82 OR prediction <= 0.3 
        ORDER BY prediction DESC
    """)
    results = cursor.fetchall()
    connection.close()

    buy = [row for row in results if row[1] >= 0.7]
    sell = [row for row in results if row[1] <= 0.3]

    # Fetch user trades
    connection = sqlite3.connect('user_trades.db')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT symbol FROM bought_stock 
        WHERE user_id = ? 
        ORDER BY timestamp DESC
    """, (user_id,))
    symbols = cursor.fetchall()
    connection.close()

    # Fetch predictions and stop-loss for each trade
    trades = []
    connection = sqlite3.connect('stock_data.db')
    cursor = connection.cursor()
    for symbol_tuple in symbols:
        symbol = symbol_tuple[0]
        cursor.execute("""
            SELECT prediction, stop_loss 
            FROM predictions 
            WHERE symbol = ?
        """, (symbol,))
        prediction_data = cursor.fetchone()
        if prediction_data:
            prediction, stop_loss = prediction_data
            trades.append({"symbol": symbol, "prediction": prediction, "stop_loss": stop_loss})
    connection.close()

    return render_template('index.html', buy=buy, sell=sell, trades=trades)


@app.route('/add_trade', methods=['POST'])
@login_required
def add_trade():
    symbol = request.form.get('symbol').upper()
    user_id = session['user_id']

    connection = sqlite3.connect('stock_data.db')
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE symbol = ?", (symbol,))
    exists = cursor.fetchone()[0]
    connection.close()

    if not exists:
        flash(f"Symbol '{symbol}' not found in predictions!", "error")
        return redirect('/')

    connection = sqlite3.connect('user_trades.db')
    cursor = connection.cursor()
    cursor.execute("INSERT INTO bought_stock (symbol, user_id) VALUES (?, ?)", (symbol, user_id))
    connection.commit()
    connection.close()

    # Fetch updated trades
    return redirect('/')

@app.route('/remove_trade', methods=['POST'])
@login_required
def remove_trade():
    symbol = request.form.get('symbol').upper()
    user_id = session['user_id']

    # Remove the symbol from the bought_stock table
    connection = sqlite3.connect('user_trades.db')
    cursor = connection.cursor()
    cursor.execute("""
        DELETE FROM bought_stock WHERE symbol = ? AND user_id = ?
    """, (symbol, user_id))
    connection.commit()
    connection.close()

    # Requery the predictions table for the remaining symbols
    connection = sqlite3.connect('user_trades.db')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT symbol FROM bought_stock WHERE user_id = ?
    """, (user_id,))
    remaining_symbols = [row[0] for row in cursor.fetchall()]
    connection.close()

    # Fetch updated predictions for remaining symbols
    connection = sqlite3.connect('stock_data.db')
    cursor = connection.cursor()
    if remaining_symbols:
        placeholder = ', '.join('?' for _ in remaining_symbols)
        cursor.execute(f"""
            SELECT symbol, prediction, stop_loss
            FROM predictions
            WHERE symbol IN ({placeholder})
        """, remaining_symbols)
        updated_trades = cursor.fetchall()
    else:
        updated_trades = []  # No trades remain
    connection.close()

    # Render the updated table
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
        conn = sqlite3.connect('stock_data.db')
        # Make sure to set row_factory if needed for dict-like access:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, date, time, open_price, prediction, stop_loss, take_profit FROM predictions_15min")
        rows = cursor.fetchall()
        predictions = [dict(row) for row in rows]
        return jsonify(predictions)
    except Exception as e:
        app.logger.error(f"Error fetching predictions: {e}")
        return jsonify({"error": "Error fetching predictions"}), 500
    finally:
        conn.close()

@app.route('/full_table_15min')
@login_required
def full_table_15min():
    return render_template('full_table_15min.html')

@app.route('/predictions', methods=['GET'])
@login_required
def get_predictions():
    connection = sqlite3.connect('stock_data.db')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT symbol, date, time, open_price, prediction, stop_loss, take_profit, capital_invested
        FROM predictions
        ORDER BY prediction DESC
    """)
    rows = cursor.fetchall()
    connection.close()

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
    connection = sqlite3.connect('user_trades.db')
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subscription TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bought_stock (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    connection.commit()
    connection.close()
    app.run(debug=True)
