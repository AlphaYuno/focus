import sqlite3

def reset_focus_sessions():
    """Remove all entries from the focus_sessions table and reset the session ID counter."""
    try:
        # Connect to the database
        conn = sqlite3.connect("focus_enhancement.db")
        cursor = conn.cursor()
        
        # Get the count of records before deletion
        cursor.execute("SELECT COUNT(*) FROM focus_sessions")
        count_before = cursor.fetchone()[0]
        
        # Delete all records from the focus_sessions table
        cursor.execute("DELETE FROM focus_sessions")
        
        # Also delete related records in the allowed_apps table
        cursor.execute("DELETE FROM allowed_apps")
        
        # Reset the session_id counter to start from 1
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='focus_sessions'")
        
        # Commit the changes
        conn.commit()
        
        # Verify the reset
        cursor.execute("SELECT COUNT(*) FROM focus_sessions")
        count_after = cursor.fetchone()[0]
        
        # Close the connection
        conn.close()
        
        print(f"Successfully removed {count_before} entries from the focus_sessions table.")
        print(f"Current count: {count_after}")
        print("Session ID counter has been reset. Next session will start with ID 1.")
        
    except Exception as e:
        print(f"Error resetting focus sessions: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    # Ask for confirmation before proceeding
    confirm = input("This will delete ALL focus session data and reset the counter. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        reset_focus_sessions()
    else:
        print("Operation cancelled.")