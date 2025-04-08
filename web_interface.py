#!/usr/bin/env python
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import requests
import json
from datetime import datetime, timedelta
import os
import uuid
import pickle
import hashlib
import secrets

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for sessions
API_URL = "http://localhost:8001"  # Default to the simple API

# In-memory storage for historical predictions (in production, use a database)
history_store = {}

# Simple user database - in production, use a proper database
# Format: {'username': {'password_hash': '...', 'salt': '...', 'role': 'user|admin'}}
user_db = {}

def hash_password(password, salt=None):
    """Hash a password with a salt for secure storage"""
    if salt is None:
        salt = secrets.token_hex(16)
    # Create salted password hash
    pw_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()
    return pw_hash, salt

def verify_password(stored_password_hash, stored_salt, provided_password):
    """Verify a password against a stored hash"""
    calculated_hash, _ = hash_password(provided_password, stored_salt)
    return secrets.compare_digest(calculated_hash, stored_password_hash)

def create_default_user():
    """Create a default admin user if no users exist"""
    if not user_db:
        password_hash, salt = hash_password("admin")
        user_db["admin"] = {
            "password_hash": password_hash,
            "salt": salt,
            "role": "admin",
            "name": "Administrator"
        }
        # Save to file
        with open('user_db.pkl', 'wb') as f:
            pickle.dump(user_db, f)
        print("Created default admin user (username: admin, password: admin)")

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, redirect to home
    if 'username' in session:
        return redirect(url_for('index'))
    
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in user_db:
            stored_hash = user_db[username]['password_hash']
            stored_salt = user_db[username]['salt']
            
            if verify_password(stored_hash, stored_salt, password):
                session['username'] = username
                session['user_id'] = username  # Use username as user_id for history
                session['role'] = user_db[username]['role']
                session['name'] = user_db[username].get('name', username)
                
                # Initialize history for this user if not exists
                if username not in history_store:
                    history_store[username] = []
                
                return redirect(url_for('index'))
            else:
                error = "Invalid password"
        else:
            error = "Username not found"
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('role', None)
    session.pop('name', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Only admins can register new users
    if 'role' in session and session['role'] == 'admin':
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            name = request.form.get('name')
            role = request.form.get('role', 'user')
            
            if username in user_db:
                flash('Username already exists', 'danger')
                return redirect(url_for('register'))
            
            # Create new user
            password_hash, salt = hash_password(password)
            user_db[username] = {
                "password_hash": password_hash,
                "salt": salt,
                "role": role,
                "name": name
            }
            
            # Save to file
            with open('user_db.pkl', 'wb') as f:
                pickle.dump(user_db, f)
            
            flash(f'User {username} created successfully', 'success')
            return redirect(url_for('index'))
        
        return render_template('register.html')
    else:
        flash('Permission denied', 'danger')
        return redirect(url_for('index'))

@app.route('/')
def index():
    # Check if user is authenticated
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user info
    username = session['username']
    user_role = session['role']
    name = session['name']
    
    # Initialize history for this user if not exists
    if username not in history_store:
        history_store[username] = []
    
    # Get list of patients
    try:
        response = requests.get(f"{API_URL}/patients")
        patients = response.json()["patients"]
    except:
        patients = [570]  # Default if API is not available
        
    return render_template('index.html', 
                          patients=patients, 
                          username=username, 
                          role=user_role, 
                          name=name)

@app.route('/get_horizons', methods=['POST'])
def get_horizons():
    # Check if user is authenticated
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    patient_id = request.form.get('patient_id')
    try:
        response = requests.get(f"{API_URL}/horizons", params={"patient_id": patient_id})
        horizons = response.json()["horizons"]
        return jsonify({"horizons": horizons})
    except:
        # Default horizons if API is not available
        return jsonify({"horizons": [15, 30]})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if user is authenticated
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get form data
    data = {
        "patient_id": int(request.form.get('patient_id')),
        "glucose_value": float(request.form.get('glucose_value')),
        "glucose_diff": float(request.form.get('glucose_diff')),
        "glucose_diff_rate": float(request.form.get('glucose_diff_rate')),
        "glucose_rolling_mean_1h": float(request.form.get('glucose_rolling_mean_1h')),
        "glucose_rolling_std_1h": float(request.form.get('glucose_rolling_std_1h')),
        "hour": int(request.form.get('hour', datetime.now().hour)),
        "day_of_week": int(request.form.get('day_of_week', datetime.now().weekday())),
        # Default values for remaining fields
        "glucose_lags": [float(request.form.get('glucose_value'))] * 12,
        "insulin_dose": float(request.form.get('insulin_dose', 0)),
        "insulin_dose_1h": float(request.form.get('insulin_dose_1h', 0)),
        "insulin_dose_2h": float(request.form.get('insulin_dose_2h', 0)),
        "insulin_dose_4h": float(request.form.get('insulin_dose_4h', 0)),
        "carbs_1h": float(request.form.get('carbs_1h', 0)),
        "carbs_2h": float(request.form.get('carbs_2h', 0)),
        "carbs_4h": float(request.form.get('carbs_4h', 0))
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        prediction_result = response.json()
        
        # Format data for plotting
        now = datetime.now()
        result = {
            "prediction_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "current_glucose": prediction_result["current_glucose"],
            "predictions": prediction_result["predictions"],
            "timestamps": {},
            "features_used": prediction_result.get("features_used", {})
        }
        
        # Create time points for plotting
        result["timestamps"]["current"] = now.strftime("%H:%M")
        
        # Store all prediction times for history
        prediction_times = {}
        
        for horizon in prediction_result["predictions"]:
            minutes = int(horizon.replace("min", ""))
            future_time = now + timedelta(minutes=minutes)
            result["timestamps"][horizon] = future_time.strftime("%H:%M")
            prediction_times[horizon] = future_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save prediction to history
        if 'username' in session:
            history_entry = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "display_time": now.strftime("%H:%M"),
                "patient_id": data["patient_id"],
                "current_glucose": data["glucose_value"],
                "predictions": prediction_result["predictions"],
                "prediction_times": prediction_times,
                "inputs": {
                    "glucose_diff": data["glucose_diff"],
                    "glucose_diff_rate": data["glucose_diff_rate"],
                    "insulin_dose": data["insulin_dose"],
                    "carbs_1h": data["carbs_1h"]
                }
            }
            
            # Add to the beginning of the list (most recent first)
            history_store[session['username']].insert(0, history_entry)
            
            # Keep only the most recent 20 predictions
            if len(history_store[session['username']]) > 20:
                history_store[session['username']] = history_store[session['username']][:20]
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/history', methods=['GET'])
def get_history():
    # Check if user is authenticated
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    if session['username'] in history_store:
        history = history_store[session['username']]
        return jsonify({"history": history})
    else:
        return jsonify({"history": []})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    # Check if user is authenticated
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    if session['username'] in history_store:
        history_store[session['username']] = []
    return jsonify({"success": True})

@app.route('/users', methods=['GET'])
def get_users():
    # Only admins can view user list
    if 'role' in session and session['role'] == 'admin':
        users = []
        for username, data in user_db.items():
            users.append({
                "username": username,
                "name": data.get('name', username),
                "role": data.get('role', 'user')
            })
        return render_template('users.html', users=users)
    else:
        flash('Permission denied', 'danger')
        return redirect(url_for('index'))

@app.route('/delete_user/<username>', methods=['POST'])
def delete_user(username):
    # Only admins can delete users
    if 'role' in session and session['role'] == 'admin':
        if username == 'admin':
            flash('Cannot delete admin user', 'danger')
        elif username == session['username']:
            flash('Cannot delete your own account', 'danger')
        else:
            if username in user_db:
                del user_db[username]
                # Save to file
                with open('user_db.pkl', 'wb') as f:
                    pickle.dump(user_db, f)
                flash(f'User {username} deleted successfully', 'success')
            else:
                flash('User not found', 'danger')
    else:
        flash('Permission denied', 'danger')
    
    return redirect(url_for('get_users'))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Try to load user database
    try:
        with open('user_db.pkl', 'rb') as f:
            user_db = pickle.load(f)
    except:
        user_db = {}
        create_default_user()
    
    # Try to load history from disk if exists
    try:
        with open('history_store.pkl', 'rb') as f:
            history_store = pickle.load(f)
    except:
        history_store = {}
    
    # Save history and user data to disk on exit
    import atexit
    def save_data():
        with open('history_store.pkl', 'wb') as f:
            pickle.dump(history_store, f)
        with open('user_db.pkl', 'wb') as f:
            pickle.dump(user_db, f)
    atexit.register(save_data)
    
    app.run(debug=True, port=5001) 