import time
import psutil
import win32process
import win32gui
import win32con
import win32api
import os
import winreg
import subprocess
import pandas as pd
import re
from datetime import datetime
import pygetwindow as gw
import psutil
import time
from PyQt5.QtWidgets import QMessageBox, QApplication
import sys
import threading

import pickle

focus_data_csv = pd.read_csv('../finalised/dataset/focus_data.csv')


class AppTracker:
    def __init__(self):
        # Tracking attributes
        self.allowed_apps = []
        self.app_switch_count = 0
        self.distraction_time = 0
        self.focus_time = 0
        self.last_check_time = None
        self.last_app = None
        self.tracking = False
        
        # App detection attributes
        self.current_app = None  # Track the currently active app/window
        self.timer_started = False
        self.timer = None
        self.productive_apps = ['Google Chrome', 'VS Code', 'Sublime Text', 'Notepad++']  # List of apps allowed for focus
        
    def set_allowed_apps(self, app_list):
        """Set the list of allowed applications for the focus session."""
        self.allowed_apps = [app.lower() for app in app_list]
        
    def get_active_window_process_name(self):
        """Get the process name of the currently active window."""
        try:
            # Get the handle of the active window
            hwnd = win32gui.GetForegroundWindow()
            
            # Safety check for invalid window handle
            if hwnd == 0:
                return "No active window"
            
            # Get the window title
            try:
                window_title = win32gui.GetWindowText(hwnd)
            except:
                window_title = ""
            
            # Get the process ID
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
            except:
                return "Unknown process"
            
            # Get the process name
            try:
                process = psutil.Process(pid)
                process_name = process.name()
            except psutil.NoSuchProcess:
                return "Process not found"
            except psutil.AccessDenied:
                return "Access denied"
            except:
                return "Unknown process"
            
            # Check if it's Chrome with a specific tab
            if process_name.lower() == "chrome.exe" and window_title:
                # Limit the length of the window title to prevent buffer overflow
                max_title_length = 100
                if len(window_title) > max_title_length:
                    window_title = window_title[:max_title_length] + "..."
                
                # Remove the " - Google Chrome" suffix if present
                if " - Google Chrome" in window_title:
                    tab_title = window_title.replace(" - Google Chrome", "")
                    # Extract website name
                    website_name = self.extract_website_name(tab_title)
                    return f"Chrome: {website_name}"
                
                return f"Chrome: {window_title}"
            
            return process_name
        except Exception as e:
            print(f"Error getting active window: {str(e)}")
            return "Error"
    
    def get_start_menu_apps(self):
        """Get a list of applications from the Start Menu."""
        apps = []
        
        # Common paths for Start Menu shortcuts
        start_menu_paths = [
            os.path.join(os.environ["APPDATA"], "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ["ProgramData"], "Microsoft", "Windows", "Start Menu", "Programs")
        ]
        
        # Collect .lnk files from Start Menu
        for start_menu_path in start_menu_paths:
            if os.path.exists(start_menu_path):
                for root, dirs, files in os.walk(start_menu_path):
                    for file in files:
                        if file.endswith(".lnk"):
                            app_name = os.path.splitext(file)[0]
                            if app_name not in apps:
                                apps.append(app_name)
        
        print("Start Menu Apps:", apps)  # Debugging statement
        return sorted(apps)
    
    def get_chrome_tabs(self):
        """Get a list of open Chrome tabs."""
        chrome_tabs = []
        
        try:
            # Check if Chrome is running
            chrome_running = False
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and proc.info['name'].lower() == "chrome.exe":
                    chrome_running = True
                    break
            
            if chrome_running:
                # Get all window handles
                def enum_windows_callback(hwnd, tabs):
                    if win32gui.IsWindowVisible(hwnd):
                        window_title = win32gui.GetWindowText(hwnd)
                        try:
                            _, pid = win32process.GetWindowThreadProcessId(hwnd)
                            process = psutil.Process(pid)
                            if process.name().lower() == "chrome.exe" and window_title and " - Google Chrome" in window_title:
                                # Remove the " - Google Chrome" suffix
                                tab_title = window_title.replace(" - Google Chrome", "")
                                
                                # Extract website name from the tab title
                                website_name = self.extract_website_name(tab_title)
                                
                                tabs.append(f"Chrome: {website_name}")
                        except:
                            pass
                    return True
                
                win32gui.EnumWindows(enum_windows_callback, chrome_tabs)
        except:
            pass
        
        return chrome_tabs
    
    def extract_website_name(self, tab_title):
        """Extract the website name from a Chrome tab title."""
        # First, try to extract from the URL in the title
        url_pattern = r'https?://(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)(?:/|$)'
        url_match = re.search(url_pattern, tab_title)
        
        # Next, look for domain patterns in the title
        domain_pattern = r'(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)(?:\s|$)'
        domain_match = re.search(domain_pattern, tab_title)
        
        # Extract from URL if found
        if url_match:
            domain = url_match.group(1)
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
            
        # Extract domain if found in title
        elif domain_match:
            domain = domain_match.group(1)
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        
        # For URLs that might be in the window title but not matched by the patterns
        for known_domain in ["coursera.org", "github.com", "stackoverflow.com", "youtube.com", 
                            "google.com", "facebook.com", "twitter.com", "linkedin.com"]:
            if known_domain in tab_title.lower():
                return known_domain.split('.')[0].capitalize()
        
        # Common patterns for website titles as fallback
        # 1. "Page Title | Website Name"
        # 2. "Page Title - Website Name"
        # 3. "Website Name: Page Title"
        for separator in [' | ', ' - ', ': ']:
            parts = tab_title.split(separator)
            if len(parts) > 1:
                # For patterns 1 and 2, website name is usually the last part
                if separator in [' | ', ' - ']:
                    candidate = parts[-1].strip()
                    # Only use if it's reasonably short (likely a site name, not a page title)
                    if len(candidate) < 20:
                        return candidate
                # For pattern 3, website name is usually the first part
                elif separator == ': ':
                    candidate = parts[0].strip()
                    if len(candidate) < 20:
                        return candidate
        
        # If all else fails, return the original title (limited to 30 chars)
        if len(tab_title) > 30:
            return tab_title[:27] + "..."
        return tab_title
    
    def get_running_apps(self):
        """Get a list of currently running applications visible in Task Manager's Apps section."""
        running_apps = []
        
        try:
            # Get all running processes with window
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    process_info = proc.info
                    pid = process_info['pid']
                    
                    # Check if the process has a window (visible in Task Manager's Apps section)
                    if self.has_window(pid):
                        process_name = process_info['name']
                        # Remove the .exe extension if present
                        if process_name.lower().endswith(".exe"):
                            process_name = process_name[:-4]
                        running_apps.append(process_name)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception as e:
            print(f"Error getting running apps: {str(e)}")
        
        # Get Chrome tabs
        chrome_tabs = self.get_chrome_tabs()
        
        # Combine the lists and remove duplicates
        all_apps = running_apps + chrome_tabs
        
        return sorted(list(set(all_apps)))
    
    def has_window(self, pid):
        """Check if a process has a visible window."""
        def callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                _, process_id = win32process.GetWindowThreadProcessId(hwnd)
                if process_id == pid:
                    text = win32gui.GetWindowText(hwnd)
                    if text and len(text) > 0:  # Only consider windows with a title
                        hwnds.append(hwnd)
            return True
        
        hwnds = []
        try:
            win32gui.EnumWindows(callback, hwnds)
        except Exception:
            pass
        
        return len(hwnds) > 0
    
    def start_tracking(self):
        """Start tracking app usage and focus time."""
        self.app_switch_count = 0
        self.distraction_time = 0
        self.focus_time = 0
        self.last_check_time = datetime.now()
        self.last_app = None
        self.tracking = True
        
    def stop_tracking(self):
        """Stop tracking and return metrics."""
        self.tracking = False
        return self.app_switch_count, self.distraction_time, self.focus_time
    
    def check_current_app(self):
        """Check the currently active application and update tracking metrics."""
        if not self.tracking:
            return None
        
        try:
            current_time = datetime.now()
            
            # Safety check: ensure last_check_time is initialized
            if self.last_check_time is None:
                self.last_check_time = current_time
                return None
            
            time_diff = (current_time - self.last_check_time).total_seconds() / 60  # Convert to minutes
            
            # Safety check: ensure time_diff is reasonable (prevent negative or extremely large values)
            if time_diff < 0 or time_diff > 60:  # Cap at 60 minutes
                time_diff = 0
            
            current_app = self.get_active_window_process_name()
            
            # If the app has changed, increment the switch count
            if current_app != self.last_app and self.last_app is not None:
                self.app_switch_count += 1
                self.last_app = current_app
            elif self.last_app is None:
                self.last_app = current_app
            
            # Check if the current app is in the allowed list
            is_allowed = False
            if self.allowed_apps:  # Only check if we have allowed apps
                for allowed in self.allowed_apps:
                    if allowed and current_app and allowed.lower() in current_app.lower():
                        is_allowed = True
                        break
            
            # Update focus or distraction time
            if is_allowed:
                self.focus_time += time_diff
            else:
                self.distraction_time += time_diff
            
            self.last_check_time = current_time
            
            return current_app, is_allowed
        except Exception as e:
            print(f"Error in check_current_app: {str(e)}")
            return None 
  
    def track_app_usage(self):
        """Track app usage and detect app switching."""
        while True:
            active_app = self.get_active_app()

            # If the user switches to a non-productive app
            if active_app != self.current_app:
                if self.is_distracting(active_app):  # Check if the app is distracting
                    self.start_timer()  # Start the timer for distraction detection
                else:
                    self.current_app = active_app  # Update current app
                    print(f"Currently using: {self.current_app}")

            time.sleep(1)  # Check every second for app change

    def get_active_app(self):
        """Get the title of the currently active window (app)."""
        try:
            active_window = gw.getActiveWindow().title  # Get active window title
            return active_window
        except Exception as e:
            print(f"Error detecting active app: {e}")
            return None  # Return None if the app detection fails

    def is_distracting(self, app_name):
        """Check if the app is distracting (i.e., not in the productive apps list)."""
        for app in self.productive_apps:
            if app in app_name:  # If the app name contains any productive apps
                return False  # If it's a productive app, it's not distracting
        return True  # If it's not in the productive apps list, it's a distraction

    def start_timer(self):
        """Start a 10-second timer when a distracting app is detected."""
        if not self.timer_started:
            self.timer_started = True
            print("Timer started. 10 seconds to check if the user is distracted.")
            self.timer = time.time()  # Start the timer

            # Run the timer in a background thread to avoid blocking the app
            timer_thread = threading.Thread(target=self.run_timer)
            timer_thread.start()

    def run_timer(self):
        while time.time() - self.timer < 10:
            time.sleep(1)  # Wait for 10 seconds

        # Show the pop-up after 10 seconds of distraction
        self.show_popup()  # Corrected to call the method on the instance

    def show_popup(self):
        """Display a pop-up alert to the user about distraction."""
        app = QApplication(sys.argv)  # Initialize the PyQt application
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("You are distracting yourself!")  # Pop-up message
        msg.setWindowTitle("Distraction Alert")
        msg.setStandardButtons(QMessageBox.Ok)  # Add an OK button
        msg.buttonClicked.connect(self.popup_button_clicked)  # Connect the OK button
        msg.exec_()  # Display the pop-up window

    def popup_button_clicked(self, i):
        """Handle the button click in the pop-up."""
        print("User clicked OK. Distraction alert dismissed.")
        self.timer_started = False  # Reset the timer after dismissal
        
        
  