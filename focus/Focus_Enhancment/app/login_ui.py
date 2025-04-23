from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QFrame,
                             QStackedWidget, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon

class LoginWidget(QWidget):
    login_successful = pyqtSignal(int, str, bool)  # Signal to emit user_id, username, and remember_me when login is successful
    
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(50, 50, 50, 50)
        
        # Title
        title_label = QLabel("Focus Enhancement")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Pomodoro Timer & Focus Tracker")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("Arial", 14))
        main_layout.addWidget(subtitle_label)
        
        # Add some spacing
        main_layout.addSpacing(30)
        
        # Create stacked widget for login and register forms
        self.stacked_widget = QStackedWidget()
        
        # Login form
        login_widget = QWidget()
        login_layout = QVBoxLayout(login_widget)
        
        login_title = QLabel("Login")
        login_title.setFont(QFont("Arial", 16, QFont.Bold))
        login_title.setAlignment(Qt.AlignCenter)
        login_layout.addWidget(login_title)
        
        # Username
        username_layout = QHBoxLayout()
        username_label = QLabel("Username:")
        username_label.setFixedWidth(100)
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        login_layout.addLayout(username_layout)
        
        # Password
        password_layout = QHBoxLayout()
        password_label = QLabel("Password:")
        password_label.setFixedWidth(100)
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        login_layout.addLayout(password_layout)
        
        # Remember me checkbox
        remember_layout = QHBoxLayout()
        self.remember_checkbox = QCheckBox("Remember me")
        self.remember_checkbox.setChecked(True)  # Default to checked
        remember_layout.addWidget(self.remember_checkbox)
        remember_layout.addStretch()
        login_layout.addLayout(remember_layout)
        
        # Login button
        login_button = QPushButton("Login")
        login_button.setFixedHeight(40)
        login_button.clicked.connect(self.login)
        login_layout.addWidget(login_button)
        
        # Register link
        register_layout = QHBoxLayout()
        register_label = QLabel("Don't have an account?")
        register_button = QPushButton("Register")
        register_button.setFlat(True)
        register_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        register_layout.addWidget(register_label)
        register_layout.addWidget(register_button)
        register_layout.addStretch()
        login_layout.addLayout(register_layout)
        
        # Register form
        register_widget = QWidget()
        register_layout = QVBoxLayout(register_widget)
        
        register_title = QLabel("Register")
        register_title.setFont(QFont("Arial", 16, QFont.Bold))
        register_title.setAlignment(Qt.AlignCenter)
        register_layout.addWidget(register_title)
        
        # Username
        reg_username_layout = QHBoxLayout()
        reg_username_label = QLabel("Username:")
        reg_username_label.setFixedWidth(100)
        self.reg_username_input = QLineEdit()
        self.reg_username_input.setPlaceholderText("Choose a username")
        reg_username_layout.addWidget(reg_username_label)
        reg_username_layout.addWidget(self.reg_username_input)
        register_layout.addLayout(reg_username_layout)
        
        # Password
        reg_password_layout = QHBoxLayout()
        reg_password_label = QLabel("Password:")
        reg_password_label.setFixedWidth(100)
        self.reg_password_input = QLineEdit()
        self.reg_password_input.setPlaceholderText("Choose a password")
        self.reg_password_input.setEchoMode(QLineEdit.Password)
        reg_password_layout.addWidget(reg_password_label)
        reg_password_layout.addWidget(self.reg_password_input)
        register_layout.addLayout(reg_password_layout)
        
        # Confirm Password
        confirm_password_layout = QHBoxLayout()
        confirm_password_label = QLabel("Confirm:")
        confirm_password_label.setFixedWidth(100)
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setPlaceholderText("Confirm your password")
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        confirm_password_layout.addWidget(confirm_password_label)
        confirm_password_layout.addWidget(self.confirm_password_input)
        register_layout.addLayout(confirm_password_layout)
        
        # Register button
        register_submit_button = QPushButton("Register")
        register_submit_button.setFixedHeight(40)
        register_submit_button.clicked.connect(self.register)
        register_layout.addWidget(register_submit_button)
        
        # Login link
        login_link_layout = QHBoxLayout()
        login_link_label = QLabel("Already have an account?")
        login_link_button = QPushButton("Login")
        login_link_button.setFlat(True)
        login_link_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        login_link_layout.addWidget(login_link_label)
        login_link_layout.addWidget(login_link_button)
        login_link_layout.addStretch()
        register_layout.addLayout(login_link_layout)
        
        # Add both forms to stacked widget
        self.stacked_widget.addWidget(login_widget)
        self.stacked_widget.addWidget(register_widget)
        
        # Add stacked widget to main layout
        main_layout.addWidget(self.stacked_widget)
        
        # Set the layout
        self.setLayout(main_layout)
        
    def login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        remember_me = self.remember_checkbox.isChecked()
        
        if not username or not password:
            QMessageBox.warning(self, "Login Error", "Please enter both username and password.")
            return
        
        success, message, user_id = self.db.authenticate_user(username, password)
        
        if success:
            self.login_successful.emit(user_id, username, remember_me)
        else:
            QMessageBox.warning(self, "Login Error", message)
    
    def register(self):
        username = self.reg_username_input.text().strip()
        password = self.reg_password_input.text()
        confirm_password = self.confirm_password_input.text()
        
        if not username or not password or not confirm_password:
            QMessageBox.warning(self, "Registration Error", "Please fill in all fields.")
            return
        
        if password != confirm_password:
            QMessageBox.warning(self, "Registration Error", "Passwords do not match.")
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, "Registration Error", "Password must be at least 6 characters long.")
            return
        
        success, message = self.db.register_user(username, password)
        
        if success:
            QMessageBox.information(self, "Registration Successful", "Your account has been created. You can now log in.")
            self.stacked_widget.setCurrentIndex(0)
            self.reg_username_input.clear()
            self.reg_password_input.clear()
            self.confirm_password_input.clear()
        else:
            QMessageBox.warning(self, "Registration Error", message) 