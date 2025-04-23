# Focus Enhancement - Basic Version

This is the basic version of the Focus Enhancement application, without cloud functionality. It includes all the core features:

## Features

- **Pomodoro Timer**: A customizable timer to help you focus on tasks for set periods of time
- **Todo List**: Keep track of your tasks and mark them as complete
- **App Tracking**: Monitor which applications you use during your focus sessions
- **Statistics**: View your productivity data and track your progress
- **User Accounts**: Create an account to save your data
- **Session Persistence**: Stay logged in between application restarts

## Requirements

- Python 3.8 or higher
- PyQt5
- Other dependencies listed in requirements.txt

## Installation

1. Make sure you have Python installed
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python main.py
   ```

## Usage

1. Register a new account or log in with an existing one
2. Add tasks to your todo list
3. Select applications to track during your focus sessions
4. Start a Pomodoro timer and focus on your work
5. View your statistics to track your progress

## Files

- `main.py`: The main application entry point
- `database.py`: Database management for storing user data
- `login_ui.py`: User authentication interface
- `todo_ui.py`: Todo list interface
- `pomodoro_ui.py`: Pomodoro timer interface
- `stats_ui.py`: Statistics and data visualization interface
- `app_tracker.py`: Application usage tracking
- `session_manager.py`: User session management

## Note

This is the basic version of the application without cloud synchronization. For the cloud-enabled version, see the main project directory. 