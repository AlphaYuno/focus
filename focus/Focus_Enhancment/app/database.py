import sqlite3
import bcrypt
import os
from datetime import datetime, timedelta
import uuid
import pickle
import pandas as pd






class Database:
    def __init__(self, db_name="focus_enhancement.db"):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.initialize_database()

    def connect(self):
        """Establish a connection to the database."""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        return self.conn, self.cursor

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def initialize_database(self):
        """Create the necessary tables if they don't exist."""
        self.connect()
        
        # Create users table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create tasks table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Create focus sessions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS focus_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            task_id INTEGER,
            date TEXT NOT NULL,
            day TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            task_type TEXT,
            app_switch_count INTEGER DEFAULT 0,
            distraction_duration REAL DEFAULT 0,
            total_focus_duration REAL DEFAULT 0,
            focus_score INTEGER,
            productivity_percentage REAL,
            break_duration INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (task_id) REFERENCES tasks (task_id)
        )
        ''')
        
        # Create allowed apps table for each session
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS allowed_apps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            app_name TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES focus_sessions (session_id)
        )
        ''')
        
        # Create user sessions table for persistent login
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        self.conn.commit()
        self.close()

    def register_user(self, username, password):
        """Register a new user."""
        try:
            self.connect()
            
            # Check if username already exists
            self.cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if self.cursor.fetchone():
                self.close()
                return False, "Username already exists"
            
            # Hash the password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Insert the new user
            self.cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash.decode('utf-8'))
            )
            self.conn.commit()
            self.close()
            return True, "User registered successfully"
        except Exception as e:
            self.close()
            return False, f"Error registering user: {str(e)}"

    def authenticate_user(self, username, password):
        """Authenticate a user."""
        try:
            self.connect()
            
            # Get the user's password hash
            self.cursor.execute(
                "SELECT user_id, password_hash FROM users WHERE username = ?",
                (username,)
            )
            result = self.cursor.fetchone()
            
            if not result:
                self.close()
                return False, "Invalid username or password", None
            
            user_id, password_hash = result
            
            # Check if the password matches
            if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                self.close()
                return True, "Authentication successful", user_id
            else:
                self.close()
                return False, "Invalid username or password", None
        except Exception as e:
            self.close()
            return False, f"Error authenticating user: {str(e)}", None

    def add_task(self, user_id, title, description=""):
        """Add a new task for a user."""
        try:
            self.connect()
            self.cursor.execute(
                "INSERT INTO tasks (user_id, title, description) VALUES (?, ?, ?)",
                (user_id, title, description)
            )
            task_id = self.cursor.lastrowid
            self.conn.commit()
            self.close()
            return True, "Task added successfully", task_id
        except Exception as e:
            self.close()
            return False, f"Error adding task: {str(e)}", None

    def get_tasks(self, user_id, status="active"):
        """Get all tasks for a user with the specified status."""
        try:
            self.connect()
            self.cursor.execute(
                "SELECT task_id, title, description, created_at FROM tasks WHERE user_id = ? AND status = ? ORDER BY created_at DESC",
                (user_id, status)
            )
            tasks = self.cursor.fetchall()
            self.close()
            return tasks
        except Exception as e:
            self.close()
            return []

    def update_task_status(self, task_id, status):
        """Update the status of a task."""
        try:
            self.connect()
            if status == "completed":
                self.cursor.execute(
                    "UPDATE tasks SET status = ?, completed_at = ? WHERE task_id = ?",
                    (status, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), task_id)
                )
            else:
                self.cursor.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    (status, task_id)
                )
            self.conn.commit()
            self.close()
            return True, "Task status updated successfully"
        except Exception as e:
            self.close()
            return False, f"Error updating task status: {str(e)}"

    def update_task_details(self, task_id, title, description):
        """Update the details of a task."""
        try:
            self.connect()
            self.cursor.execute(
                "UPDATE tasks SET title = ?, description = ? WHERE task_id = ?",
                (title, description, task_id)
            )
            self.conn.commit()
            self.close()
            return True, "Task updated successfully"
        except Exception as e:
            self.close()
            return False, f"Error updating task: {str(e)}"

    def start_focus_session(self, user_id, task_id, task_type):
        """Start a new focus session."""
        try:
            self.connect()
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            day = now.strftime("%A")
            start_time = now.strftime("%H:%M:%S")
            
            self.cursor.execute(
                """INSERT INTO focus_sessions 
                   (user_id, task_id, date, day, start_time, task_type) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, task_id, date, day, start_time, task_type)
            )
            session_id = self.cursor.lastrowid
            self.conn.commit()
            self.close()
            return True, "Focus session started", session_id
        except Exception as e:
            self.close()
            return False, f"Error starting focus session: {str(e)}", None

    def add_allowed_app(self, session_id, app_name):
        """Add an allowed app for a focus session."""
        try:
            self.connect()
            self.cursor.execute(
                "INSERT INTO allowed_apps (session_id, app_name) VALUES (?, ?)",
                (session_id, app_name)
            )
            self.conn.commit()
            self.close()
            return True, "Allowed app added successfully"
        except Exception as e:
            self.close()
            return False, f"Error adding allowed app: {str(e)}"

    def get_allowed_apps(self, session_id):
        """Get all allowed apps for a focus session."""
        try:
            self.connect()
            self.cursor.execute(
                "SELECT app_name FROM allowed_apps WHERE session_id = ?",
                (session_id,)
            )
            apps = [row[0] for row in self.cursor.fetchall()]
            self.close()
            return apps
        except Exception as e:
            self.close()
            return []

    def end_focus_session(self, session_id, app_switch_count, distraction_duration, 
                         total_focus_duration, focus_score, break_duration=0):
        """End a focus session and record the results."""
        try:
            self.connect()
            now = datetime.now()
            end_time = now.strftime("%H:%M:%S")
            
            # Calculate productivity percentage
            if total_focus_duration + distraction_duration > 0:
                productivity = (total_focus_duration / (total_focus_duration + distraction_duration)) * 100
            else:
                productivity = 0
            
            self.cursor.execute(
                """UPDATE focus_sessions SET 
                   end_time = ?, 
                   app_switch_count = ?, 
                   distraction_duration = ?, 
                   total_focus_duration = ?, 
                   focus_score = ?, 
                   productivity_percentage = ?,
                   break_duration = ?
                   WHERE session_id = ?""",
                (end_time, app_switch_count, distraction_duration, 
                 total_focus_duration, focus_score, productivity, break_duration, session_id)
            )
            self.conn.commit()
            self.close()
            return True, "Focus session ended successfully"
        except Exception as e:
            self.close()
            return False, f"Error ending focus session: {str(e)}"

    def get_user_sessions(self, user_id, limit=10):
        """Get the most recent focus sessions for a user."""
        try:
            self.connect()
            self.cursor.execute(
                """SELECT session_id, date, day, start_time, end_time, task_type, 
                   app_switch_count, distraction_duration, total_focus_duration, 
                   focus_score, productivity_percentage, break_duration
                   FROM focus_sessions 
                   WHERE user_id = ? 
                   ORDER BY date DESC, start_time DESC 
                   LIMIT ?""",
                (user_id, limit)
            )
            sessions = self.cursor.fetchall()
            self.close()
            return sessions
        except Exception as e:
            self.close()
            return []

    def get_user_sessions_by_period(self, user_id, period="all"):
        """Get focus sessions for a user within a specific time period.
        
        Args:
            user_id: The user ID
            period: The time period ("day", "week", "month", "year", or "all")
            
        Returns:
            A list of session tuples
        """
        try:
            self.connect()
            
            # Calculate the date range based on the period
            today = datetime.now().date()
            start_date = None
            
            if period == "day":
                start_date = today
            elif period == "week":
                # Get the Monday of the current week
                start_date = today - timedelta(days=today.weekday())
            elif period == "month":
                # Get the first day of the current month
                start_date = today.replace(day=1)
            elif period == "year":
                # Get the first day of the current year
                start_date = today.replace(month=1, day=1)
            
            # Build the query
            query = """SELECT session_id, date, day, start_time, end_time, task_type, 
                      app_switch_count, distraction_duration, total_focus_duration, 
                      focus_score, productivity_percentage, break_duration
                      FROM focus_sessions 
                      WHERE user_id = ?"""
            params = [user_id]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.strftime("%Y-%m-%d"))
            
            query += " ORDER BY date ASC, start_time ASC"
            
            self.cursor.execute(query, params)
            sessions = self.cursor.fetchall()
            self.close()
            return sessions
        except Exception as e:
            self.close()
            print(f"Error getting sessions by period: {str(e)}")
            return []

    def create_user_session(self, user_id, username, days_valid=30):
        """Create a persistent session for a user.
        
        Args:
            user_id: The user's ID
            username: The user's username
            days_valid: Number of days the session should remain valid
            
        Returns:
            Tuple of (success, message, session_token)
        """
        try:
            self.connect()
            
            # Generate a unique session token
            session_token = str(uuid.uuid4())
            
            # Calculate expiration date
            now = datetime.now()
            expires_at = now + timedelta(days=days_valid)
            
            # Insert the session
            self.cursor.execute(
                """INSERT INTO user_sessions 
                   (session_token, user_id, username, created_at, expires_at) 
                   VALUES (?, ?, ?, ?, ?)""",
                (session_token, user_id, username, now.isoformat(), expires_at.isoformat())
            )
            
            self.conn.commit()
            self.close()
            return True, "Session created successfully", session_token
        except Exception as e:
            self.close()
            return False, f"Error creating session: {str(e)}", None
    
    def get_session(self, session_token):
        """Get user information from a session token.
        
        Args:
            session_token: The session token to validate
            
        Returns:
            Tuple of (success, message, user_id, username)
        """
        try:
            self.connect()
            
            # Get the session
            self.cursor.execute(
                """SELECT user_id, username, expires_at 
                   FROM user_sessions 
                   WHERE session_token = ?""",
                (session_token,)
            )
            
            result = self.cursor.fetchone()
            
            if not result:
                self.close()
                return False, "Invalid session", None, None
            
            user_id, username, expires_at = result
            
            # Check if the session has expired
            now = datetime.now()
            expiration = datetime.fromisoformat(expires_at)
            
            if now > expiration:
                # Delete the expired session
                self.delete_session(session_token)
                self.close()
                return False, "Session expired", None, None
            
            self.close()
            return True, "Valid session", user_id, username
        except Exception as e:
            self.close()
            return False, f"Error validating session: {str(e)}", None, None
    
    def delete_session(self, session_token):
        """Delete a user session.
        
        Args:
            session_token: The session token to delete
            
        Returns:
            Tuple of (success, message)
        """
        try:
            self.connect()
            
            # Delete the session
            self.cursor.execute(
                "DELETE FROM user_sessions WHERE session_token = ?",
                (session_token,)
            )
            
            self.conn.commit()
            self.close()
            return True, "Session deleted successfully"
        except Exception as e:
            self.close()
            return False, f"Error deleting session: {str(e)}"
    
    def delete_all_user_sessions(self, user_id):
        """Delete all sessions for a specific user.
        
        Args:
            user_id: The user ID to delete sessions for
            
        Returns:
            Tuple of (success, message)
        """
        try:
            self.connect()
            
            # Delete all sessions for the user
            self.cursor.execute(
                "DELETE FROM user_sessions WHERE user_id = ?",
                (user_id,)
            )
            
            self.conn.commit()
            self.close()
            return True, "All sessions deleted successfully"
        except Exception as e:
            self.close()
            return False, f"Error deleting sessions: {str(e)}"
    
    def update_break_duration(self, session_id, actual_break_duration):
        """Update the break duration with the actual time the break ran for.
        
        Args:
            session_id: The ID of the focus session
            actual_break_duration: The actual duration of the break in minutes
            
        Returns:
            A tuple (success, message)
        """
        try:
            self.connect()
            
            self.cursor.execute(
                """UPDATE focus_sessions SET 
                   break_duration = ?
                   WHERE session_id = ?""",
                (actual_break_duration, session_id)
            )
            self.conn.commit()
            self.close()
            return True, "Break duration updated successfully"
        except Exception as e:
            self.close()
            return False, f"Error updating break duration: {str(e)}" 