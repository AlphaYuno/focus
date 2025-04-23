import sys
import threading
import pygetwindow as gw
import pickle
import pandas as pd



from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QStackedWidget, QTabWidget, 
                             QPushButton, QLabel, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

from database import Database
from login_ui import LoginWidget
from todo_ui import TodoWidget
from pomodoro_ui import PomodoroWidget
from stats_ui import StatsWidget
from suggestions_ui import SuggestionsUI
from app_tracker import AppTracker
from session_manager import SessionManager

# Dark theme colors
DARK_PRIMARY = "#1e1e1e"
DARK_SECONDARY = "#2d2d2d"
DARK_TERTIARY = "#3e3e3e"
DARK_TEXT = "#ffffff"
ACCENT_COLOR = "#3daee9"
WARNING_COLOR = "#f39c12"
ERROR_COLOR = "#e74c3c"
SUCCESS_COLOR = "#2ecc71"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = Database()
        self.app_tracker = AppTracker()
        self.session_manager = SessionManager()
        self.user_id = None
        self.username = None
        self.session_token = None
        self.init_ui()
        
        # Check for existing session
        self.check_existing_session()
        
    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Focus Enhancement - Pomodoro Timer")
        self.setMinimumSize(1000, 700)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create stacked widget for login and main app
        self.stacked_widget = QStackedWidget()
        
        # Create login widget
        self.login_widget = LoginWidget(self.db)
        self.login_widget.login_successful.connect(self.on_login_successful)
        
        # Add widgets to stacked widget
        self.stacked_widget.addWidget(self.login_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(self.stacked_widget)
        
        # Show the window
        self.show()
        
        # Start app tracker
        self.app_tracker_timer = QTimer(self)
        self.app_tracker_timer.timeout.connect(self.track_apps)
        self.app_tracker_timer.start(60000)  # Track every minute
    
    def check_existing_session(self):
        """Check if there's an existing session and log in automatically if found."""
        session_token = self.session_manager.load_session()
        
        if session_token:
            success, message, user_id, username = self.db.get_session(session_token)
            
            if success:
                self.session_token = session_token
                self.on_login_successful(user_id, username, False, create_session=False)
            else:
                # Session is invalid or expired, clear it
                self.session_manager.clear_session()
    
    def on_login_successful(self, user_id, username, remember_me=True, create_session=True):
        """Handle successful login."""
        self.user_id = user_id
        self.username = username
        
        # Create a persistent session if needed
        if create_session and remember_me:
            success, message, session_token = self.db.create_user_session(user_id, username)
            if success:
                self.session_token = session_token
                self.session_manager.save_session(session_token)
        
        # Create main app widget
        main_app_widget = QWidget()
        main_layout = QVBoxLayout(main_app_widget)
        
        # Header with user info and logout button
        header_layout = QHBoxLayout()
        
        welcome_label = QLabel(f"Welcome, {username}!")
        welcome_label.setFont(QFont("Arial", 14, QFont.Bold))
        
        logout_button = QPushButton("Logout")
        logout_button.clicked.connect(self.logout)
        
        header_layout.addWidget(welcome_label)
        header_layout.addStretch()
        header_layout.addWidget(logout_button)
        
        main_layout.addLayout(header_layout)
        
        # Create content layout (todo list on left, pomodoro on right)
        content_layout = QHBoxLayout()
        
        # Create todo widget
        self.todo_widget = TodoWidget(self.db, self.user_id)
        
        # Create pomodoro widget
        self.pomodoro_widget = PomodoroWidget(self.db, self.user_id, self.app_tracker)
        
        # Create stats widget
        self.stats_widget = StatsWidget(self.db, self.user_id)
        
        # Create suggestions widget
        self.suggestions_widget = SuggestionsUI(self.db, self.user_id)
        
        # Create tab widget for pomodoro, stats, and suggestions
        tab_widget = QTabWidget()
        tab_widget.addTab(self.pomodoro_widget, "Pomodoro Timer")
        tab_widget.addTab(self.stats_widget, "Statistics")
        tab_widget.addTab(self.suggestions_widget, "Suggestions")
        
        # Add widgets to content layout
        content_layout.addWidget(self.todo_widget, 1)  # 1/3 of width
        content_layout.addWidget(tab_widget, 2)        # 2/3 of width
        
        main_layout.addLayout(content_layout)
        
        # Add main app widget to stacked widget
        self.stacked_widget.addWidget(main_app_widget)
        
        # Switch to main app
        self.stacked_widget.setCurrentIndex(1)
    
    def logout(self):
        """Log out the current user."""
        # Confirm logout
        reply = QMessageBox.question(
            self, 
            "Confirm Logout", 
            "Are you sure you want to log out?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Delete the session
            if self.session_token:
                self.db.delete_session(self.session_token)
                self.session_manager.clear_session()
                self.session_token = None
            
            # Reset user info
            self.user_id = None
            self.username = None
            
            # Remove main app widget from stacked widget
            main_app_widget = self.stacked_widget.widget(1)
            self.stacked_widget.removeWidget(main_app_widget)
            main_app_widget.deleteLater()
            
            # Switch back to login
            self.stacked_widget.setCurrentIndex(0)
            
            # Clear login fields
            self.login_widget.username_input.clear()
            self.login_widget.password_input.clear()
    
    def track_apps(self):
        if self.user_id:
            # check_current_app returns None if tracking is not active
            result = self.app_tracker.check_current_app()
            if result:  # Only proceed if we got a valid result
                current_app, is_allowed = result
                # Update the UI if needed with the current app information
                if hasattr(self, 'pomodoro_widget'):
                    self.pomodoro_widget.update_current_app(current_app, is_allowed)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for a modern look
    
    # Apply dark theme palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(DARK_PRIMARY))
    dark_palette.setColor(QPalette.WindowText, QColor(DARK_TEXT))
    dark_palette.setColor(QPalette.Base, QColor(DARK_SECONDARY))
    dark_palette.setColor(QPalette.AlternateBase, QColor(DARK_TERTIARY))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(DARK_TEXT))
    dark_palette.setColor(QPalette.ToolTipText, QColor(DARK_TEXT))
    dark_palette.setColor(QPalette.Text, QColor(DARK_TEXT))
    dark_palette.setColor(QPalette.Button, QColor(DARK_SECONDARY))
    dark_palette.setColor(QPalette.ButtonText, QColor(DARK_TEXT))
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(ACCENT_COLOR))
    dark_palette.setColor(QPalette.Highlight, QColor(ACCENT_COLOR))
    dark_palette.setColor(QPalette.HighlightedText, QColor(DARK_TEXT))
    
    # Apply the dark palette
    app.setPalette(dark_palette)
    
    # Set stylesheet for additional styling
    app.setStyleSheet(f"""
        QWidget {{
            background-color: {DARK_PRIMARY};
            color: {DARK_TEXT};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {DARK_TERTIARY};
            background-color: {DARK_SECONDARY};
        }}
        
        QTabBar::tab {{
            background-color: {DARK_TERTIARY};
            color: {DARK_TEXT};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {ACCENT_COLOR};
        }}
        
        QPushButton {{
            background-color: {DARK_TERTIARY};
            color: {DARK_TEXT};
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        QPushButton:hover {{
            background-color: {ACCENT_COLOR};
        }}
        
        QLineEdit, QTextEdit, QComboBox {{
            background-color: {DARK_SECONDARY};
            color: {DARK_TEXT};
            border: 1px solid {DARK_TERTIARY};
            border-radius: 4px;
            padding: 6px;
        }}
        
        QProgressBar {{
            border: 1px solid {DARK_TERTIARY};
            border-radius: 4px;
            text-align: center;
        }}
        
        QProgressBar::chunk {{
            background-color: {ACCENT_COLOR};
            width: 10px;
        }}
        
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
        }}
        
        QCheckBox::indicator:unchecked {{
            border: 1px solid {DARK_TERTIARY};
            background-color: {DARK_SECONDARY};
        }}
        
        QCheckBox::indicator:checked {{
            border: 1px solid {ACCENT_COLOR};
            background-color: {ACCENT_COLOR};
        }}
        
        QTableView {{
            background-color: {DARK_SECONDARY};
            alternate-background-color: {DARK_TERTIARY};
            selection-background-color: {ACCENT_COLOR};
            gridline-color: {DARK_TERTIARY};
            border: 1px solid {DARK_TERTIARY};
        }}
        
        QHeaderView::section {{
            background-color: {DARK_TERTIARY};
            color: {DARK_TEXT};
            padding: 4px;
            border: 1px solid {DARK_PRIMARY};
        }}
        
        QScrollBar:vertical {{
            border: none;
            background: {DARK_TERTIARY};
            width: 12px;
            margin: 0px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {DARK_SECONDARY};
            min-height: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            border: none;
            background: {DARK_TERTIARY};
            height: 12px;
            margin: 0px;
        }}
        
        QScrollBar::handle:horizontal {{
            background: {DARK_SECONDARY};
            min-width: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
    """)
    
    window = MainWindow()
    sys.exit(app.exec_()) 
from app_tracker import AppTracker

# Create an instance of the AppTracker
app_tracker = AppTracker()

# Run the app tracker in a separate thread to avoid blocking the main application
app_tracker_thread = threading.Thread(target=app_tracker.track_app_usage)
app_tracker_thread.daemon = True  # This makes the thread stop when the main program stops
app_tracker_thread.start()