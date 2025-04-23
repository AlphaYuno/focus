import os
import json
from pathlib import Path

class SessionManager:
    """Manages user session persistence."""
    
    def __init__(self):
        """Initialize the session manager."""
        # Create the app data directory if it doesn't exist
        self.app_data_dir = os.path.join(os.path.expanduser("~"), ".focus_enhancement")
        os.makedirs(self.app_data_dir, exist_ok=True)
        
        # Path to the session file
        self.session_file = os.path.join(self.app_data_dir, "session.json")
    
    def save_session(self, session_token):
        """Save a session token to the local system.
        
        Args:
            session_token: The session token to save
        """
        try:
            with open(self.session_file, 'w') as f:
                json.dump({"session_token": session_token}, f)
            return True
        except Exception as e:
            print(f"Error saving session: {str(e)}")
            return False
    
    def load_session(self):
        """Load a session token from the local system.
        
        Returns:
            The session token if found, None otherwise
        """
        try:
            if not os.path.exists(self.session_file):
                return None
            
            with open(self.session_file, 'r') as f:
                data = json.load(f)
                return data.get("session_token")
        except Exception as e:
            print(f"Error loading session: {str(e)}")
            return None
    
    def clear_session(self):
        """Clear the saved session from the local system."""
        try:
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
            return True
        except Exception as e:
            print(f"Error clearing session: {str(e)}")
            return False 