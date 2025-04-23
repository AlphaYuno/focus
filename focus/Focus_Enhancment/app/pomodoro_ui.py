from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QFrame,
                             QSpinBox, QSlider, QCheckBox, QListWidget, QListWidgetItem,
                             QProgressBar, QComboBox, QGroupBox, QRadioButton, QButtonGroup,
                             QFormLayout, QDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import time
from datetime import datetime, timedelta
import pandas as pd
import pickle
with open('../finalised/Best_Day.pkl', 'rb') as f:
    best_day_model = pickle.load(f)

with open('../finalised/Best_Length.pkl', 'rb') as f:
    best_length_model = pickle.load(f)

with open('../finalised/Best_Time.pkl', 'rb') as f:
    best_time_model = pickle.load(f)


class PomodoroWidget(QWidget):
    session_ended = pyqtSignal(int, int, float, float, int)  # Signal to emit session data when ended
    
    def __init__(self, db, user_id, app_tracker):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.app_tracker = app_tracker
        self.session_id = None
        self.task_id = None
        self.task_type = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.remaining_seconds = 0
        self.total_seconds = 0
        self.start_time = None
        self.init_ui()
        
        # Load initial data
        self.load_tasks()
        self.load_running_apps()
    
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        self.title_label = QLabel("Pomodoro Timer")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)
        
        # Task info section
        task_group = QGroupBox("Current Task")
        task_layout = QVBoxLayout()
        task_layout.setSpacing(8)  # Add consistent spacing
        
        # Task selection dropdown
        task_selection_layout = QFormLayout()  # Use FormLayout for better alignment
        self.task_combo = QComboBox()
        self.task_combo.setMinimumWidth(250)
        self.task_combo.currentIndexChanged.connect(self.on_task_selected)
        task_selection_layout.addRow("Select Task:", self.task_combo)
        
        # Task type selection
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["Coding", "Writing", "Studying", "Reading", "Meeting", "Other"])
        task_selection_layout.addRow("Task Type:", self.task_type_combo)
        
        # Add layouts to task section
        task_layout.addLayout(task_selection_layout)
        
        # Refresh tasks button
        refresh_button_layout = QHBoxLayout()
        refresh_tasks_button = QPushButton("Refresh Tasks")
        refresh_tasks_button.clicked.connect(self.load_tasks)
        refresh_tasks_button.setFixedHeight(30)
        refresh_button_layout.addStretch()
        refresh_button_layout.addWidget(refresh_tasks_button)
        refresh_button_layout.addStretch()
        task_layout.addLayout(refresh_button_layout)
        
        task_group.setLayout(task_layout)
        main_layout.addWidget(task_group)
        
        # App selection section in its own group box
        app_group = QGroupBox("App Selection")
        app_layout = QVBoxLayout()
        app_layout.setSpacing(8)  # Add consistent spacing
        
        # Search bar for apps
        self.app_search_bar = QLineEdit()
        self.app_search_bar.setPlaceholderText("Search apps...")
        self.app_search_bar.textChanged.connect(self.filter_apps)
        app_layout.addWidget(self.app_search_bar)
        
        # App selection label with better description
        app_selection_label = QLabel("Select apps you'll be using during this session:")
        app_layout.addWidget(app_selection_label)
        
        # App list and buttons in horizontal layout
        app_list_layout = QHBoxLayout()
        
        # App list with better size
        self.app_list = QListWidget()
        self.app_list.setSelectionMode(QListWidget.MultiSelection)
        self.app_list.setMinimumHeight(150)
        app_list_layout.addWidget(self.app_list, 4)  # 80% of width
        
        # App buttons in vertical layout
        app_buttons_layout = QVBoxLayout()
        app_buttons_layout.setSpacing(5)
        
        refresh_apps_button = QPushButton("Refresh Apps")
        refresh_apps_button.clicked.connect(self.load_running_apps)
        refresh_apps_button.setFixedHeight(30)
        
        select_all_apps_button = QPushButton("Select All")
        select_all_apps_button.clicked.connect(self.select_all_apps)
        select_all_apps_button.setFixedHeight(30)
        
        clear_apps_button = QPushButton("Clear Selection")
        clear_apps_button.clicked.connect(self.clear_app_selection)
        clear_apps_button.setFixedHeight(30)
        
        app_buttons_layout.addWidget(refresh_apps_button)
        app_buttons_layout.addWidget(select_all_apps_button)
        app_buttons_layout.addWidget(clear_apps_button)
        app_buttons_layout.addStretch()  # Push buttons to the top
        
        app_list_layout.addLayout(app_buttons_layout, 1)  # 20% of width
        
        app_layout.addLayout(app_list_layout)
        
        app_group.setLayout(app_layout)
        main_layout.addWidget(app_group)
        
        # Timer display
        timer_group = QGroupBox("Timer")
        timer_layout = QVBoxLayout()
        timer_layout.setSpacing(10)  # Add consistent spacing
        
        # Duration setting in a horizontal layout
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration:"))
        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 120)
        self.duration_input.setValue(25)
        self.duration_input.setSingleStep(5)
        self.duration_input.setSuffix(" min")
        duration_layout.addWidget(self.duration_input)
        duration_layout.addStretch()
        timer_layout.addLayout(duration_layout)
        
        # Timer display with larger font
        self.time_display = QLabel("00:00")
        self.time_display.setFont(QFont("Arial", 60, QFont.Bold))
        self.time_display.setAlignment(Qt.AlignCenter)
        self.time_display.setStyleSheet("color: #ffffff;")  # White color for dark theme
        timer_layout.addWidget(self.time_display)
        
        # Progress bar with better styling
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        timer_layout.addWidget(self.progress_bar)
        
        # Timer controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)  # Add spacing between buttons
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_timer)
        self.start_button.setFixedHeight(40)
        self.start_button.setMinimumWidth(120)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_timer)
        self.pause_button.setFixedHeight(40)
        self.pause_button.setMinimumWidth(120)
        self.pause_button.setEnabled(False)
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_timer)
        self.stop_button.setFixedHeight(40)
        self.stop_button.setMinimumWidth(120)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        controls_layout.addStretch()  # Center the buttons
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()  # Center the buttons
        
        timer_layout.addLayout(controls_layout)
        
        timer_group.setLayout(timer_layout)
        main_layout.addWidget(timer_group)
        
        # Set the layout
        self.setLayout(main_layout)
    
    def load_running_apps(self):
        """Load the list of running applications."""
        self.app_list.clear()
        running_apps = self.app_tracker.get_running_apps()
        
        # Group apps by category
        chrome_tabs = []
        other_apps = []
        
        for app in running_apps:
            if app.startswith("Chrome:"):
                chrome_tabs.append(app)
            else:
                other_apps.append(app)
        
        # Add Chrome tabs with a header
        if chrome_tabs:
            chrome_header = QListWidgetItem("--- Chrome Tabs ---")
            chrome_header.setFlags(Qt.ItemIsEnabled)  # Make it non-selectable
            chrome_header.setBackground(self.app_list.palette().highlight())
            chrome_header.setForeground(self.app_list.palette().highlightedText())
            self.app_list.addItem(chrome_header)
            
            for tab in chrome_tabs:
                self.app_list.addItem(tab)
        
        # Add other apps with a header
        if other_apps:
            apps_header = QListWidgetItem("--- Applications ---")
            apps_header.setFlags(Qt.ItemIsEnabled)  # Make it non-selectable
            apps_header.setBackground(self.app_list.palette().highlight())
            apps_header.setForeground(self.app_list.palette().highlightedText())
            self.app_list.addItem(apps_header)
            
            for app in other_apps:
                self.app_list.addItem(app)
    
    def load_tasks(self):
        """Load active tasks from the database."""
        self.task_combo.clear()
        self.task_combo.addItem("Select a task...", None)  # Default option
        
        tasks = self.db.get_tasks(self.user_id, status="active")
        
        for task_id, title, description, created_at in tasks:
            self.task_combo.addItem(title, task_id)
    
    def on_task_selected(self, index):
        """Handle task selection from dropdown."""
        if index <= 0:  # Skip the default "Select a task..." option
            self.task_id = None
            return
            
        self.task_id = self.task_combo.itemData(index)
        self.task_type = self.task_type_combo.currentText()
    
    def set_task(self, task_id, task_type):
        """Set the current task for the Pomodoro session."""
        self.task_id = task_id
        self.task_type = task_type
        
        # Find and select the task in the dropdown
        for i in range(self.task_combo.count()):
            if self.task_combo.itemData(i) == task_id:
                self.task_combo.setCurrentIndex(i)
                break
        
        # Set the task type
        self.task_type_combo.setCurrentText(task_type)
    
    def start_timer(self):
        """Start the Pomodoro timer."""
        try:
            # Check if a task is selected
            if not self.task_id:
                msg_box = QMessageBox()
                msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("No Task Selected")
                msg_box.setText("Please select a task from the dropdown first.")
                msg_box.exec_()
                return
            
            # Check if apps are selected
            selected_items = self.app_list.selectedItems()
            selected_apps = []
            
            for item in selected_items:
                # Skip header items
                if item.text().startswith("---"):
                    continue
                selected_apps.append(item.text())
                
            if not selected_apps:
                msg_box = QMessageBox()
                msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("No Apps Selected")
                msg_box.setText("Please select at least one app that you'll be using during this session.")
                msg_box.exec_()
                return
            
            # Get task type
            self.task_type = self.task_type_combo.currentText()
            
            # Get duration
            minutes = self.duration_input.value()
            self.total_seconds = minutes * 60
            self.remaining_seconds = self.total_seconds
            
            # Update UI
            self.update_time_display()
            self.progress_bar.setValue(0)
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.duration_input.setEnabled(False)
            self.app_list.setEnabled(False)
            
            # Update title based on whether this is a break or focus session
            break_index = self.task_combo.findText("Break Time")
            in_break_mode = break_index >= 0 and self.task_combo.currentIndex() == break_index
            if in_break_mode:
                self.title_label.setText(f"Break Time - {minutes} Minutes")
            else:
                self.title_label.setText(f"Focus Session - {minutes} Minutes")
            
            # Start the timer
            self.timer.start(1000)  # Update every second
            self.start_time = datetime.now()
            
            # Start app tracking
            self.app_tracker.set_allowed_apps(selected_apps)
            self.app_tracker.start_tracking()
            
            # Create session in database
            success, message, session_id = self.db.start_focus_session(
                self.user_id, 
                self.task_id, 
                self.task_type
            )
            
            if success:
                self.session_id = session_id
                # Add allowed apps to database
                for app in selected_apps:
                    self.db.add_allowed_app(session_id, app)
            else:
                QMessageBox.warning(self, "Error", f"Failed to start session: {message}")
                self.stop_timer()
        except Exception as e:
            print(f"Error starting timer: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred while starting the timer: {str(e)}")
            # Try to safely stop the timer
            try:
                self.timer.stop()
                self.app_tracker.stop_tracking()
            except:
                pass
            # Reset UI
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.duration_input.setEnabled(True)
            self.app_list.setEnabled(True)
    
    def pause_timer(self):
        """Pause or resume the Pomodoro timer."""
        try:
            if self.timer.isActive():
                # Pause the timer
                self.timer.stop()
                self.pause_button.setText("Resume")
                
                # Store the current tracking state and pause tracking
                if hasattr(self, 'session_id') and self.session_id:
                    self._tracking_was_active = True
                    # Get current tracking data before pausing
                    app_switch_count, distraction_time, focus_time = self.app_tracker.stop_tracking()
                    # Store the tracking data for when we resume
                    self._paused_tracking_data = {
                        "app_switch_count": app_switch_count,
                        "distraction_time": distraction_time,
                        "focus_time": focus_time
                    }
            else:
                # Resume the timer
                self.timer.start(1000)
                self.pause_button.setText("Pause")
                
                # Resume tracking if it was active before
                if hasattr(self, '_tracking_was_active') and self._tracking_was_active:
                    # Restart tracking
                    self.app_tracker.start_tracking()
                    
                    # Restore the tracking data if we have it
                    if hasattr(self, '_paused_tracking_data'):
                        self.app_tracker.app_switch_count = self._paused_tracking_data["app_switch_count"]
                        self.app_tracker.distraction_time = self._paused_tracking_data["distraction_time"]
                        self.app_tracker.focus_time = self._paused_tracking_data["focus_time"]
        except Exception as e:
            print(f"Error in pause_timer: {str(e)}")
    
    def stop_timer(self):
        """Stop the Pomodoro timer."""
        try:
            self.timer.stop()
            
            # Reset title
            self.title_label.setText("Pomodoro Timer")
            
            # Check if we're in break mode
            break_index = self.task_combo.findText("Break Time")
            in_break_mode = break_index >= 0 and self.task_combo.currentIndex() == break_index
            
            if in_break_mode:
                # Just exit the break without showing feedback
                self.exit_break()
                return
            
            # Stop app tracking
            try:
                app_switch_count, distraction_time, focus_time = self.app_tracker.stop_tracking()
            except Exception as e:
                print(f"Error stopping app tracking: {str(e)}")
                app_switch_count, distraction_time, focus_time = 0, 0, 0
            
            # Update UI
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.pause_button.setText("Pause")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stop")  # Reset button text
            self.duration_input.setEnabled(True)
            self.app_list.setEnabled(True)
            
            # Remove the "Break Time" option if it exists
            if break_index >= 0:
                self.task_combo.removeItem(break_index)
                # Restore previous task selection if available
                if hasattr(self, 'previous_task_index') and self.previous_task_index >= 0:
                    self.task_combo.setCurrentIndex(self.previous_task_index)
            
            # Show feedback dialog instead of embedded form
            if self.session_id:
                # Store session data for feedback submission
                self.session_data = {
                    "app_switch_count": app_switch_count,
                    "distraction_time": distraction_time,
                    "focus_time": focus_time
                }
                
                # Show the feedback dialog
                self.show_feedback_dialog()
        except Exception as e:
            print(f"Error stopping timer: {str(e)}")
            # Make sure UI is reset
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.pause_button.setText("Pause")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stop")
            self.duration_input.setEnabled(True)
            self.app_list.setEnabled(True)
            self.time_display.setText("00:00")
            self.progress_bar.setValue(0)
    
    def update_timer(self):
        """Update the timer display and progress."""
        try:
            if self.remaining_seconds > 0:
                self.remaining_seconds -= 1
                self.update_time_display()
                
                # Update progress bar
                progress = 100 - int((self.remaining_seconds / self.total_seconds) * 100)
                self.progress_bar.setValue(progress)
                
                # Check current app every 5 seconds, but only if we're in a focus session (not a break)
                if self.remaining_seconds % 5 == 0 and self.session_id is not None:
                    result = self.app_tracker.check_current_app()
                    # Only process the result if it's not None
                    if result is not None:
                        current_app, is_allowed = result
            else:
                # Timer finished
                self.timer.stop()
                
                # Check if we're in break mode
                break_index = self.task_combo.findText("Break Time")
                in_break_mode = break_index >= 0 and self.task_combo.currentIndex() == break_index
                
                if in_break_mode:
                    # Calculate actual break duration (full duration since timer completed)
                    if hasattr(self, 'planned_break_duration'):
                        self.actual_break_duration = self.planned_break_duration
                        
                        # Update the database with the actual break duration
                        if hasattr(self, 'last_session_id') and self.last_session_id:
                            self.db.update_break_duration(self.last_session_id, self.actual_break_duration)
                    
                    # Create message box with stay-on-top flag
                    msg_box = QMessageBox()
                    msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.setWindowTitle("Break Ended")
                    msg_box.setText("Your break has ended. Ready to start a new focus session?")
                    msg_box.exec_()
                    
                    self.exit_break()
                else:
                    # Create message box with stay-on-top flag
                    msg_box = QMessageBox()
                    msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.setWindowTitle("Time's Up!")
                    msg_box.setText("Your Pomodoro session has ended.")
                    msg_box.exec_()
                    
                    self.stop_timer()
        except Exception as e:
            print(f"Error in update_timer: {str(e)}")
            # Try to safely stop the timer to prevent further errors
            try:
                self.timer.stop()
            except:
                pass
    
    def update_time_display(self):
        """Update the time display label."""
        minutes = self.remaining_seconds // 60
        seconds = self.remaining_seconds % 60
        self.time_display.setText(f"{minutes:02d}:{seconds:02d}")
    
    def show_feedback_dialog(self):
        """Show a dialog to collect feedback about the session."""
        try:
            # Create a session feedback dialog that stays on top
            dialog = SessionFeedbackDialog(self)
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
            
            result = dialog.exec_()
            
            if result == QDialog.Accepted:
                focus_score = dialog.focus_slider.value()
                break_option = dialog.break_options.checkedId()
                
                # Set break duration based on user selection
                break_duration = 0  # Default to 0 if user exits
                if break_option == 1:  # Take a break
                    break_duration = dialog.custom_break_time
                
                # End session in database
                if self.session_id:
                    try:
                        success, message = self.db.end_focus_session(
                            self.session_id,
                            self.session_data["app_switch_count"],
                            self.session_data["distraction_time"],
                            self.session_data["focus_time"],
                            focus_score,
                            break_duration  # Pass the break duration to the database
                        )
                        
                        if success:
                            # Store the session ID for updating with actual break duration later
                            self.last_session_id = self.session_id
                        else:
                            msg_box = QMessageBox()
                            msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                            msg_box.setIcon(QMessageBox.Warning)
                            msg_box.setWindowTitle("Error")
                            msg_box.setText(f"Failed to save session data: {message}")
                            msg_box.exec_()
                    except Exception as e:
                        print(f"Error ending focus session: {str(e)}")
                        msg_box = QMessageBox()
                        msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                        msg_box.setIcon(QMessageBox.Warning)
                        msg_box.setWindowTitle("Error")
                        msg_box.setText(f"Failed to save session data: {str(e)}")
                        msg_box.exec_()
                
                # Reset UI
                self.time_display.setText("00:00")
                self.progress_bar.setValue(0)
                
                # Start break timer if selected
                if break_option == 1:  # Take a break
                    self.start_break(dialog.custom_break_time)
        except Exception as e:
            print(f"Error showing feedback dialog: {str(e)}")
            # Reset UI
            self.time_display.setText("00:00")
            self.progress_bar.setValue(0)
    
    def start_break(self, minutes):
        """Start a break timer."""
        try:
            self.total_seconds = minutes * 60
            self.remaining_seconds = self.total_seconds
            self.update_time_display()
            
            # Store the planned break duration and start time for calculating actual duration
            self.planned_break_duration = minutes
            self.break_start_time = datetime.now()
            
            # Update UI
            # Update title for break
            self.title_label.setText(f"Break Time - {minutes} Minutes")
            
            # Temporarily add a break option to the dropdown
            current_index = self.task_combo.currentIndex()
            self.task_combo.addItem("Break Time", -1)
            self.task_combo.setCurrentIndex(self.task_combo.count() - 1)
            self.task_type_combo.setCurrentText("Break")
            
            # Disable start button, enable pause and stop
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            
            # Change stop button text to "Exit Break"
            self.stop_button.setText("Exit Break")
            
            # Start the timer
            self.timer.start(1000)
            self.session_id = None  # No session tracking during break
            
            # Store the previous task index to restore after break
            self.previous_task_index = current_index
            
            # Ensure app tracker is not tracking during break
            self.app_tracker.stop_tracking()
        except Exception as e:
            print(f"Error starting break: {str(e)}")
            # Reset UI
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stop")
            self.duration_input.setEnabled(True)
            self.app_list.setEnabled(True)
            self.time_display.setText("00:00")
            self.progress_bar.setValue(0)
    
    def exit_break(self):
        """Exit break mode without showing feedback."""
        try:
            # Calculate actual break duration
            if hasattr(self, 'break_start_time'):
                break_end_time = datetime.now()
                actual_break_duration = (break_end_time - self.break_start_time).total_seconds() / 60
                # Round to nearest minute
                self.actual_break_duration = round(actual_break_duration)
                
                # Update the database with the actual break duration
                if hasattr(self, 'last_session_id') and self.last_session_id:
                    try:
                        self.db.update_break_duration(self.last_session_id, self.actual_break_duration)
                    except Exception as e:
                        print(f"Error updating break duration: {str(e)}")
            
            # Update UI
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.pause_button.setText("Pause")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stop")  # Reset button text
            self.duration_input.setEnabled(True)
            self.app_list.setEnabled(True)
            
            # Reset timer display
            self.time_display.setText("00:00")
            self.progress_bar.setValue(0)
            
            # Remove the "Break Time" option if it exists
            break_index = self.task_combo.findText("Break Time")
            if break_index >= 0:
                self.task_combo.removeItem(break_index)
                # Restore previous task selection if available
                if hasattr(self, 'previous_task_index') and self.previous_task_index >= 0:
                    self.task_combo.setCurrentIndex(self.previous_task_index)
            
            # Ensure we're not tracking apps after a break
            self.session_id = None
        except Exception as e:
            print(f"Error exiting break: {str(e)}")
            # Reset UI to a safe state
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.pause_button.setText("Pause")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stop")
            self.duration_input.setEnabled(True)
            self.app_list.setEnabled(True)
            self.time_display.setText("00:00")
            self.progress_bar.setValue(0)
            self.session_id = None
    
    def select_all_apps(self):
        """Select all apps in the app list."""
        self.app_list.selectAll()

    def clear_app_selection(self):
        """Clear the selection in the app list."""
        self.app_list.clearSelection()

    def filter_apps(self, query):
        """Filter the app list based on the search query."""
        query = query.lower()
        self.app_list.clear()
        running_apps = self.app_tracker.get_running_apps()
        
        # Group apps by category
        chrome_tabs = []
        other_apps = []
        
        for app in running_apps:
            if query in app.lower():
                if app.startswith("Chrome:"):
                    chrome_tabs.append(app)
                else:
                    other_apps.append(app)
        
        # Add Chrome tabs with a header
        if chrome_tabs:
            chrome_header = QListWidgetItem("--- Chrome Tabs ---")
            chrome_header.setFlags(Qt.ItemIsEnabled)  # Make it non-selectable
            chrome_header.setBackground(self.app_list.palette().highlight())
            chrome_header.setForeground(self.app_list.palette().highlightedText())
            self.app_list.addItem(chrome_header)
            
            for tab in chrome_tabs:
                self.app_list.addItem(tab)
        
        # Add other apps with a header
        if other_apps:
            apps_header = QListWidgetItem("--- Applications ---")
            apps_header.setFlags(Qt.ItemIsEnabled)  # Make it non-selectable
            apps_header.setBackground(self.app_list.palette().highlight())
            apps_header.setForeground(self.app_list.palette().highlightedText())
            self.app_list.addItem(apps_header)
            
            for app in other_apps:
                self.app_list.addItem(app)

    def update_current_app(self, current_app, is_allowed):
        """Update the UI with information about the currently active app.
        
        Args:
            current_app (str): The name of the currently active application
            is_allowed (bool): Whether the app is in the allowed list for the session
        """
        # Handle the case when current_app is None (error or not tracking)
        if current_app is None:
            return
            
        try:
            # If we're in an active session, we could show a warning if using a non-allowed app
            if self.timer.isActive() and not is_allowed and current_app:
                # You could show a temporary notification or change a status indicator
                # For now, we'll just print to console for debugging
                print(f"Warning: Using non-allowed app: {current_app}")
        except Exception as e:
            print(f"Error in update_current_app: {str(e)}")

class SessionFeedbackDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session Feedback")
        self.setMinimumWidth(400)
        self.setStyleSheet("background-color: #2d2d2d; color: white;")
        self.custom_break_time = 5  # Default custom break time
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Session Completed!")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Focus level slider
        focus_label = QLabel("Rate your focus level (1-10):")
        layout.addWidget(focus_label)
        
        self.focus_slider = QSlider(Qt.Horizontal)
        self.focus_slider.setRange(1, 10)
        self.focus_slider.setValue(5)
        self.focus_slider.setTickPosition(QSlider.TicksBelow)
        self.focus_slider.setTickInterval(1)
        
        self.focus_value_label = QLabel("5")
        self.focus_slider.valueChanged.connect(lambda v: self.focus_value_label.setText(str(v)))
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("1"))
        slider_layout.addWidget(self.focus_slider)
        slider_layout.addWidget(QLabel("10"))
        slider_layout.addWidget(self.focus_value_label)
        
        layout.addLayout(slider_layout)
        
        # Break options
        break_label = QLabel("What would you like to do next?")
        layout.addWidget(break_label)
        
        self.break_options = QButtonGroup()
        
        # Exit option
        exit_radio = QRadioButton("Exit session")
        self.break_options.addButton(exit_radio, 0)
        layout.addWidget(exit_radio)
        
        # Take a break option
        break_radio = QRadioButton("Take a break")
        break_radio.setChecked(True)
        self.break_options.addButton(break_radio, 1)
        layout.addWidget(break_radio)
        
        # Custom break time input (initially hidden)
        self.break_time_widget = QWidget()
        break_time_layout = QHBoxLayout()
        break_time_layout.setContentsMargins(20, 0, 0, 0)
        
        break_time_layout.addWidget(QLabel("Break duration:"))
        
        self.break_time_input = QSpinBox()
        self.break_time_input.setRange(1, 60)
        self.break_time_input.setValue(5)
        self.break_time_input.setSingleStep(1)
        self.break_time_input.setSuffix(" min")
        self.break_time_input.valueChanged.connect(self.update_custom_break_time)
        
        break_time_layout.addWidget(self.break_time_input)
        break_time_layout.addStretch()
        
        self.break_time_widget.setLayout(break_time_layout)
        layout.addWidget(self.break_time_widget)
        
        # Connect radio buttons to show/hide break time input
        self.break_options.buttonClicked.connect(self.toggle_break_time_input)
        
        # Submit button
        button_layout = QHBoxLayout()
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.accept)
        submit_button.setFixedHeight(40)
        submit_button.setMinimumWidth(150)
        submit_button.setStyleSheet("""
            QPushButton {
                background-color: #3daee9;
                color: white;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(submit_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Initialize the visibility of break time input
        self.toggle_break_time_input()
    
    def toggle_break_time_input(self):
        """Show or hide the break time input based on selected option"""
        selected_id = self.break_options.checkedId()
        self.break_time_widget.setVisible(selected_id == 1)  # Show only if "Take a break" is selected
    
    def update_custom_break_time(self, value):
        """Update the custom break time value"""
        self.custom_break_time = value
        self.break_time_input.setValue(value) 