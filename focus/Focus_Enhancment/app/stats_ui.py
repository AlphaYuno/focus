from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QHeaderView, QGroupBox, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from datetime import datetime, timedelta
import pandas as pd 
import pickle
focus_data = pd.read_csv('../finalised/dataset/focus_data.csv')


class StatsWidget(QWidget):
    def __init__(self, db, user_id):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Focus Statistics")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Session history table
        history_group = QGroupBox("Recent Sessions")
        history_layout = QVBoxLayout()
        
        self.session_table = QTableWidget()
        self.session_table.setColumnCount(12)
        self.session_table.setHorizontalHeaderLabels([
            "Session ID", "Date", "Day", "Start Time", "End Time", 
            "Task Type", "App Switches", "Distraction (min)", 
            "Focus (min)", "Focus Score", "Productivity %", "Break (min)"
        ])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.session_table)
        
        history_group.setLayout(history_layout)
        main_layout.addWidget(history_group)
        
        # Charts section
        charts_group = QGroupBox("Analytics")
        charts_layout = QVBoxLayout()
        
        # Chart type selector
        chart_selector_layout = QHBoxLayout()
        chart_selector_layout.addWidget(QLabel("Chart Type:"))
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Focus Score vs Time",
            "Focus vs. Distraction Time",
            "Productivity by Time of Day",
            "Focus Score Trend Over Days"
        ])
        self.chart_type_combo.currentIndexChanged.connect(self.update_chart)
        chart_selector_layout.addWidget(self.chart_type_combo)
        
        # Time period selector for Focus Score Trend
        self.time_period_widget = QWidget()
        self.time_period_layout = QHBoxLayout(self.time_period_widget)
        self.time_period_layout.addWidget(QLabel("Time Period:"))
        
        self.time_period_combo = QComboBox()
        self.time_period_combo.addItems(["Day", "Week", "Month", "Year"])
        self.time_period_combo.currentIndexChanged.connect(self.update_chart)
        self.time_period_layout.addWidget(self.time_period_combo)
        self.time_period_widget.setVisible(False)  # Initially hidden
        
        charts_layout.addLayout(chart_selector_layout)
        charts_layout.addWidget(self.time_period_widget)
        
        # Chart canvas with custom event handling
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)  # Ensure enough height for the chart
        
        # Connect mouse events for zooming and panning
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Variables for panning
        self.is_panning = False
        self.pan_start_x = None
        self.pan_start_y = None
        
        charts_layout.addWidget(self.canvas)
        
        # Add help text for interactive features
        help_text = QLabel("Tip: Use Ctrl + Mouse Scroll to zoom in/out. Click and drag to pan around the graph.")
        help_text.setStyleSheet("color: #666; font-style: italic;")
        help_text.setWordWrap(True)
        charts_layout.addWidget(help_text)
        
        # Reset view button
        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_view)
        charts_layout.addWidget(reset_button)
        
        charts_group.setLayout(charts_layout)
        main_layout.addWidget(charts_group)
        
        # Refresh button
        refresh_button = QPushButton("Refresh Data")
        refresh_button.clicked.connect(self.load_data)
        main_layout.addWidget(refresh_button)
        
        # Set the layout
        self.setLayout(main_layout)
        
        # Load initial data
        self.load_data()
    
    def on_scroll(self, event):
        """Handle mouse scroll events for zooming."""
        # Only zoom if Ctrl key is pressed
        if event.guiEvent.modifiers() & Qt.ControlModifier:
            ax = self.figure.axes[0]
            if not ax:
                return
                
            # Get the current x and y limits
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Calculate zoom factor
            factor = 0.9 if event.button == 'up' else 1.1
            
            # Calculate new limits centered on mouse position
            x_range = (x_max - x_min) * factor
            y_range = (y_max - y_min) * factor
            
            # Center on mouse position
            if event.xdata is not None and event.ydata is not None:
                x_center = event.xdata
                y_center = event.ydata
            else:
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
            
            # Set new limits
            ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
            ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
            
            # Redraw the canvas
            self.canvas.draw()
    
    def on_press(self, event):
        """Handle mouse button press events for panning."""
        if event.button == 1:  # Left mouse button
            self.is_panning = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata
    
    def on_release(self, event):
        """Handle mouse button release events."""
        self.is_panning = False
    
    def on_motion(self, event):
        """Handle mouse motion events for panning."""
        if self.is_panning and event.xdata is not None and event.ydata is not None:
            ax = self.figure.axes[0]
            if not ax or self.pan_start_x is None or self.pan_start_y is None:
                return
                
            # Calculate the distance moved
            dx = self.pan_start_x - event.xdata
            dy = self.pan_start_y - event.ydata
            
            # Get the current limits
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Set new limits
            ax.set_xlim([x_min + dx, x_max + dx])
            ax.set_ylim([y_min + dy, y_max + dy])
            
            # Redraw the canvas
            self.canvas.draw()
    
    def reset_view(self):
        """Reset the view to the original limits."""
        self.update_chart()  # Simply redraw the chart with default limits
    
    def load_data(self):
        """Load session data from the database."""
        sessions = self.db.get_user_sessions(self.user_id, limit=100)  # Increased limit for better analysis
        
        # Update table
        self.session_table.setRowCount(len(sessions))
        
        for row, session in enumerate(sessions):
            (session_id, date, day, start_time, end_time, task_type, 
             app_switch_count, distraction_duration, total_focus_duration, 
             focus_score, productivity_percentage, break_duration) = session
            
            # Format data for display
            if productivity_percentage is not None:
                productivity_str = f"{productivity_percentage:.1f}%"
            else:
                productivity_str = "N/A"
                
            # Add data to table
            self.session_table.setItem(row, 0, QTableWidgetItem(str(session_id)))
            self.session_table.setItem(row, 1, QTableWidgetItem(date))
            self.session_table.setItem(row, 2, QTableWidgetItem(day))
            self.session_table.setItem(row, 3, QTableWidgetItem(start_time))
            self.session_table.setItem(row, 4, QTableWidgetItem(end_time or "N/A"))
            self.session_table.setItem(row, 5, QTableWidgetItem(task_type or "N/A"))
            self.session_table.setItem(row, 6, QTableWidgetItem(str(app_switch_count)))
            self.session_table.setItem(row, 7, QTableWidgetItem(f"{distraction_duration:.1f}"))
            self.session_table.setItem(row, 8, QTableWidgetItem(f"{total_focus_duration:.1f}"))
            
            # Color-code focus score
            focus_item = QTableWidgetItem(str(focus_score) if focus_score is not None else "N/A")
            if focus_score is not None:
                # Set text color to black for better contrast with background
                focus_item.setForeground(QColor(0, 0, 0))
                focus_item.setFont(QFont("Arial", 9, QFont.Bold))
                
                if focus_score >= 8:
                    focus_item.setBackground(QColor(100, 255, 100))  # Brighter green
                elif focus_score >= 5:
                    focus_item.setBackground(QColor(255, 255, 100))  # Brighter yellow
                else:
                    focus_item.setBackground(QColor(255, 100, 100))  # Brighter red
            else:
                # Make N/A more visible
                focus_item.setForeground(QColor(255, 255, 255))  # White text
                focus_item.setFont(QFont("Arial", 9, QFont.Bold))
            self.session_table.setItem(row, 9, focus_item)
            
            # Color-code productivity
            productivity_item = QTableWidgetItem(productivity_str)
            if productivity_percentage is not None:
                # Set text color to black for better contrast with background
                productivity_item.setForeground(QColor(0, 0, 0))
                productivity_item.setFont(QFont("Arial", 9, QFont.Bold))
                
                if productivity_percentage >= 80:
                    productivity_item.setBackground(QColor(100, 255, 100))  # Brighter green
                elif productivity_percentage >= 50:
                    productivity_item.setBackground(QColor(255, 255, 100))  # Brighter yellow
                else:
                    productivity_item.setBackground(QColor(255, 100, 100))  # Brighter red
            else:
                # Make N/A more visible
                productivity_item.setForeground(QColor(255, 255, 255))  # White text
                productivity_item.setFont(QFont("Arial", 9, QFont.Bold))
            self.session_table.setItem(row, 10, productivity_item)
            
            # Add break duration with better formatting
            if break_duration is not None and break_duration > 0:
                break_item = QTableWidgetItem(f"{break_duration}")
                # Use blue color for break duration to make it stand out
                break_item.setForeground(QColor(0, 120, 215))  # Microsoft blue
                break_item.setFont(QFont("Arial", 9, QFont.Bold))
            else:
                break_item = QTableWidgetItem("0")
                # Use gray for zero break duration
                break_item.setForeground(QColor(128, 128, 128))  # Gray
            self.session_table.setItem(row, 11, break_item)
        
        # Update chart
        self.update_chart()
    
    def update_chart(self):
        """Update the chart based on the selected chart type."""
        chart_type = self.chart_type_combo.currentText()
        
        # Show/hide time period selector based on chart type
        self.time_period_widget.setVisible(chart_type == "Focus Score vs Time")
        
        # Clear the figure
        self.figure.clear()
        
        # Configure figure for dark theme
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2d2d2d')  # Dark background
        self.figure.patch.set_facecolor('#2d2d2d')  # Dark background
        
        # Set text color to white for better visibility
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        # Set grid color to a subtle gray
        ax.grid(True, linestyle='--', alpha=0.3, color='#888888')
        
        # Get sessions data
        sessions = self.db.get_user_sessions(self.user_id, limit=100)  # Increased limit for better analysis
        
        if not sessions:
            ax.text(0.5, 0.5, "No data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, color='white', fontsize=14)
            self.canvas.draw()
            return
        
        if chart_type == "Focus Score vs Time":
            time_period = self.time_period_combo.currentText()
            
            if time_period == "Day":
                # Group sessions by hour
                hour_data = {}
                for s in sessions:
                    if s[3] and s[9] is not None:  # Start time and focus score
                        try:
                            hour = int(s[3].split(':')[0])
                            if hour not in hour_data:
                                hour_data[hour] = []
                            hour_data[hour].append(s[9])  # Focus score
                        except (ValueError, IndexError):
                            continue
                
                if hour_data:
                    hours = sorted(hour_data.keys())
                    avg_scores = [sum(hour_data[h]) / len(hour_data[h]) for h in hours]
                    
                    # Plot with enhanced visibility
                    bars = ax.bar(hours, avg_scores, color='#3daee9', width=0.7)
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.1f}', ha='center', va='bottom', color='white')
                    
                    ax.set_title('Focus Score vs Time (Day)')
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Focus Score (0-10)')
                    ax.set_xticks(hours)
                    ax.set_xticklabels([f"{h}:00" for h in hours])
                else:
                    ax.text(0.5, 0.5, "No data available for focus score by hour", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, color='white', fontsize=14)
                    ax.set_title('Focus Score vs Time (Day)')
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Focus Score (0-10)')
            
            elif time_period == "Month":
                # Group sessions by day of month
                day_data = {}
                for s in sessions:
                    if s[1] and s[9] is not None:  # Date and focus score
                        try:
                            day = int(s[1].split('-')[2])
                            if day not in day_data:
                                day_data[day] = []
                            day_data[day].append(s[9])  # Focus score
                        except (ValueError, IndexError):
                            continue
                
                if day_data:
                    days = sorted(day_data.keys())
                    avg_scores = [sum(day_data[d]) / len(day_data[d]) for d in days]
                    
                    # Plot with enhanced visibility
                    bars = ax.bar(days, avg_scores, color='#3daee9', width=0.7)
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.1f}', ha='center', va='bottom', color='white')
                    
                    ax.set_title('Average Focus Score by Day (Month)')
                    ax.set_xlabel('Day of Month')
                    ax.set_ylabel('Average Focus Score (0-10)')
                    ax.set_xticks(days)
                else:
                    ax.text(0.5, 0.5, "No data available for focus score by day", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, color='white', fontsize=14)
                    ax.set_title('Average Focus Score by Day (Month)')
                    ax.set_xlabel('Day of Month')
                    ax.set_ylabel('Average Focus Score (0-10)')
            
            elif time_period == "Year":
                # Group sessions by month
                month_data = {}
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                
                for s in sessions:
                    if s[1] and s[9] is not None:  # Date and focus score
                        try:
                            month = int(s[1].split('-')[1])
                            if month not in month_data:
                                month_data[month] = []
                            month_data[month].append(s[9])  # Focus score
                        except (ValueError, IndexError):
                            continue
                
                if month_data:
                    months = sorted(month_data.keys())
                    avg_scores = [sum(month_data[m]) / len(month_data[m]) for m in months]
                    
                    # Plot with enhanced visibility
                    bars = ax.bar([month_names[m-1] for m in months], avg_scores, color='#3daee9', width=0.7)
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.1f}', ha='center', va='bottom', color='white')
                    
                    ax.set_title('Average Focus Score by Month (Year)')
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Average Focus Score (0-10)')
                else:
                    ax.text(0.5, 0.5, "No data available for focus score by month", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, color='white', fontsize=14)
                    ax.set_title('Average Focus Score by Month (Year)')
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Average Focus Score (0-10)')
        
        elif chart_type == "Focus vs. Distraction Time":
            # Extract data
            focus_times = [s[8] for s in sessions if s[8] is not None]
            distraction_times = [s[7] for s in sessions if s[7] is not None]
            
            if focus_times and distraction_times:  # Only plot if we have data
                # Create chart
                labels = ['Focus Time', 'Distraction Time']
                sizes = [sum(focus_times), sum(distraction_times)]
                colors = ['#66b3ff', '#ff9999']
                
                # Create a simpler pie chart without explode and shadow to avoid buffer overflow
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                                 autopct='%1.1f%%', shadow=False, startangle=90)
                
                # Set label text to white
                for text in texts:
                    text.set_color('white')
                    text.set_fontweight('bold')
                
                # Set percentage text to white
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                ax.set_title('Focus vs. Distraction Time Distribution')
            else:
                ax.text(0.5, 0.5, "No data available for pie chart", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, color='white', fontsize=14)
                ax.set_title('Focus vs. Distraction Time Distribution')
        
        elif chart_type == "Productivity by Time of Day":
            # Group sessions by hour
            hour_data = {}
            for s in sessions:
                if s[3] and s[9] is not None:  # Start time and focus score
                    try:
                        hour = int(s[3].split(':')[0])
                        if hour not in hour_data:
                            hour_data[hour] = []
                        hour_data[hour].append(s[9])  # Focus score
                    except (ValueError, IndexError):
                        continue
            
            if hour_data:
                hours = sorted(hour_data.keys())
                avg_productivity = [sum(hour_data[h]) / len(hour_data[h]) for h in hours]
                
                # Plot with enhanced visibility
                bars = ax.bar(hours, avg_productivity, color='#3daee9', width=0.7)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.1f}', ha='center', va='bottom', color='white')
                
                ax.set_title('Productivity by Time of Day')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Average Focus Score (0-10)')
                ax.set_xticks(hours)
                ax.set_xticklabels([f"{h}:00" for h in hours])
            else:
                ax.text(0.5, 0.5, "No data available for productivity by time of day",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, color='white', fontsize=14)
                ax.set_title('Productivity by Time of Day')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Average Focus Score (0-10)')
        
        elif chart_type == "Focus Score Trend Over Days":
            # Group sessions by date
            date_data = {}
            for s in sessions:
                if s[1] and s[9] is not None:  # Date and focus score
                    date = s[1]
                    if date not in date_data:
                        date_data[date] = []
                    date_data[date].append(s[9])  # Focus score
            
            if date_data:
                dates = sorted(date_data.keys())
                avg_scores = [sum(date_data[d]) / len(date_data[d]) for d in dates]
                
                # Plot with enhanced visibility
                ax.plot(range(len(dates)), avg_scores, 'o-', color='#3daee9', linewidth=2, markersize=8)
                
                # Add value labels next to points
                for i, score in enumerate(avg_scores):
                    ax.text(i, score + 0.1, f'{score:.1f}', ha='center', va='bottom', color='white')
                
                ax.set_title('Focus Score Trend Over Days')
                ax.set_xlabel('Date')
                ax.set_ylabel('Average Focus Score (0-10)')
                
                # Show only a subset of dates if there are many
                if len(dates) > 10:
                    step = len(dates) // 10
                    ax.set_xticks(range(0, len(dates), step))
                    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)])
                else:
                    ax.set_xticks(range(len(dates)))
                    ax.set_xticklabels(dates)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, "No data available for focus score trend",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, color='white', fontsize=14)
                ax.set_title('Focus Score Trend Over Days')
                ax.set_xlabel('Date')
                ax.set_ylabel('Average Focus Score (0-10)')
        
        self.canvas.draw() 

    def create_chart_widget(self):
        """Create the chart widget."""
        chart_widget = QWidget()
        chart_layout = QVBoxLayout()
        
        # Chart type selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Chart Type:"))
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Focus Score vs Time",
            "Productivity by Time of Day",
            "Focus Score Trend Over Days"
        ])
        self.chart_type_combo.currentIndexChanged.connect(self.update_chart)
        selector_layout.addWidget(self.chart_type_combo)
        
        # Time period selector for Focus Score Trend
        self.time_period_widget = QWidget()
        time_period_layout = QHBoxLayout(self.time_period_widget)
        time_period_layout.setContentsMargins(0, 0, 0, 0)
        
        time_period_layout.addWidget(QLabel("Time Period:"))
        self.time_period_combo = QComboBox()
        self.time_period_combo.addItems(["Day", "Month", "Year"])
        self.time_period_combo.currentIndexChanged.connect(self.update_chart)
        time_period_layout.addWidget(self.time_period_combo)
        
        selector_layout.addWidget(self.time_period_widget)
        selector_layout.addStretch()
        
        chart_layout.addLayout(selector_layout)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 4), dpi=100)
        
        # Set figure background to match dark theme
        self.figure.patch.set_facecolor('#2d2d2d')
        
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        
        chart_widget.setLayout(chart_layout)
        return chart_widget 