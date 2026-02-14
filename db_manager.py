# db_manager.py
# SQLite Database Manager for User Authentication & Session Persistence

import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, Dict, Tuple
import os

DATABASE_FILE = "users.db"


def get_db_connection():
    """Get SQLite database connection."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables on first run."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    
    # Sessions table (for persistence)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            df_uploaded BLOB,
            df_processed BLOB,
            predictions BLOB,
            data_source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def user_exists(username: str) -> bool:
    """Check if user exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def email_exists(email: str) -> bool:
    """Check if email exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def register_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    if user_exists(username):
        return False, "❌ Username already exists"
    if email_exists(email):
        return False, "❌ Email already registered"
    if len(password) < 6:
        return False, "❌ Password must be at least 6 characters"
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        conn.close()
        return True, "✅ Account created successfully! Please login."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[int]]:
    """Authenticate user. Returns (success, user_id)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    password_hash = hash_password(password)
    
    cursor.execute(
        "SELECT id FROM users WHERE username = ? AND password_hash = ?",
        (username, password_hash)
    )
    user = cursor.fetchone()
    
    if user:
        user_id = user["id"]
        # Update last login
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now(), user_id)
        )
        conn.commit()
        conn.close()
        return True, user_id
    
    conn.close()
    return False, None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user info by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def save_session_data(user_id: int, session_token: str, data: Dict) -> bool:
    """Save session data (pickled DataFrames, predictions, etc)."""
    import pickle
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        df_uploaded = pickle.dumps(data.get("df_uploaded"))
        df_processed = pickle.dumps(data.get("df_processed"))
        predictions = pickle.dumps(data.get("predictions"))
        data_source = data.get("data_source", "upload")
        
        cursor.execute(
            """INSERT OR REPLACE INTO sessions 
               (user_id, session_token, df_uploaded, df_processed, predictions, data_source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, session_token, df_uploaded, df_processed, predictions, data_source)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving session: {e}")
        return False


def load_session_data(user_id: int) -> Optional[Dict]:
    """Load session data for user."""
    import pickle
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT df_uploaded, df_processed, predictions, data_source 
               FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1""",
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "df_uploaded": pickle.loads(row["df_uploaded"]),
                "df_processed": pickle.loads(row["df_processed"]),
                "predictions": pickle.loads(row["predictions"]),
                "data_source": row["data_source"]
            }
    except Exception as e:
        print(f"Error loading session: {e}")
    
    return None
