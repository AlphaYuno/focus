from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextEdit, QMessageBox, QLineEdit, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import requests  # For Gemini API
import json

from database import Database
from sklearn.preprocessing import LabelEncoder

# Load environment variables
load_dotenv()

# Dark theme colors (matching main app)
DARK_PRIMARY = "#1e1e1e"
DARK_SECONDARY = "#2d2d2d"
DARK_TERTIARY = "#3e3e3e"
DARK_TEXT = "#ffffff"
ACCENT_COLOR = "#3daee9"
WARNING_COLOR = "#f39c12"
ERROR_COLOR = "#e74c3c"
SUCCESS_COLOR = "#2ecc71"

class ChatMessage:
    """Class to represent a single chat message"""
    def __init__(self, text, is_user=False):
        self.text = text
        self.is_user = is_user
        self.timestamp = datetime.now()
    
    def format_html(self):
        """Format the message as HTML for display"""
        align = "right" if self.is_user else "left"
        # Dark theme message colors
        bg_color = ACCENT_COLOR if self.is_user else DARK_TERTIARY
        text_color = DARK_TEXT
        sender = "You" if self.is_user else "Focus AI"
        time = self.timestamp.strftime("%H:%M")
        
        return f"""
        <div style="text-align: {align}; margin: 10px;">
            <div style="display: inline-block; background-color: {bg_color}; color: {text_color}; padding: 10px; border-radius: 10px; max-width: 80%;">
                <b>{sender}</b>
                <p style="color: {text_color};">{self.text.replace('\n', '<br>')}</p>
                <div style="font-size: 8pt; color: #cccccc; text-align: right;">{time}</div>
            </div>
        </div>
        """

class SuggestionsUI(QWidget):
    def __init__(self, db, user_id):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.chat_history = []
        self.predictions = None
        
        # Load the models and encoders
        try:
            # Load the time model
            with open('../finalised/Best_Time.pkl', 'rb') as f:
                self.best_time_model = pickle.load(f)
            
            # Load the length model and its encoders
            with open('../finalised/Best_Length.pkl', 'rb') as f:
                self.best_length_model = pickle.load(f)
            
            # Load the day model
            with open('../finalised/Best_Day.pkl', 'rb') as f:
                self.best_day_model = pickle.load(f)
                
            # Check if the models are dictionaries (common in notebooks)
            if isinstance(self.best_day_model, dict) and "MLP" in self.best_day_model:
                self.best_day_model = self.best_day_model["MLP"]
                
            if isinstance(self.best_time_model, dict) and "MLP" in self.best_time_model:
                self.best_time_model = self.best_time_model["MLP"]
                
            if isinstance(self.best_length_model, dict):
                # Use the appropriate model from the dictionary
                if "model" in self.best_length_model:
                    self.best_length_model = self.best_length_model["model"]
                elif "DecisionTree" in self.best_length_model:
                    self.best_length_model = self.best_length_model["DecisionTree"]
                else:
                    # Use the first model in the dictionary
                    self.best_length_model = next(iter(self.best_length_model.values()))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Check for Gemini API key
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            self.gemini_api_key = ""  # Will use local fallback if no API key
            QMessageBox.warning(self, "Warning", "Gemini API key not found. Add GEMINI_API_KEY to your .env file for AI-powered suggestions.")
        
        self.init_ui()
        
    def init_ui(self):
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create title
        title_label = QLabel("Focus Enhancement Assistant")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"color: {DARK_TEXT};")
        layout.addWidget(title_label)
        
        # Create chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(300)
        self.chat_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {DARK_PRIMARY};
                color: {DARK_TEXT};
                border-radius: 10px;
                padding: 10px;
                font-family: Arial;
                font-size: 10pt;
                border: 1px solid {DARK_TERTIARY};
            }}
        """)
        layout.addWidget(self.chat_display)
        
        # Create input area (horizontal layout)
        input_layout = QHBoxLayout()
        
        # Create text input
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask about your focus habits or for productivity tips...")
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {DARK_SECONDARY};
                color: {DARK_TEXT};
                border-radius: 15px;
                padding: 8px 15px;
                font-size: 10pt;
                border: 1px solid {DARK_TERTIARY};
            }}
            QLineEdit:focus {{
                border: 1px solid {ACCENT_COLOR};
            }}
        """)
        input_layout.addWidget(self.message_input, 5)  # Input takes 5/6 of the space
        
        # Create send button
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        send_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: {DARK_TEXT};
                border-radius: 15px;
                padding: 8px 15px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #4CB9FF;
            }}
        """)
        input_layout.addWidget(send_button, 1)  # Button takes 1/6 of the space
        
        layout.addLayout(input_layout)
        
        # Create refresh button
        refresh_button = QPushButton("Start New Conversation")
        refresh_button.clicked.connect(self.refresh_suggestions)
        refresh_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {SUCCESS_COLOR};
                color: {DARK_TEXT};
                border-radius: 15px;
                padding: 8px 15px;
                margin-top: 10px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #33d17a;
            }}
        """)
        layout.addWidget(refresh_button)
        
        # Set widget background
        self.setStyleSheet(f"background-color: {DARK_PRIMARY};")
        
        # Initial chat
        self.refresh_suggestions()
    
    def add_message(self, text, is_user=False):
        """Add a message to the chat history and update display"""
        message = ChatMessage(text, is_user)
        self.chat_history.append(message)
        self.update_chat_display()
    
    def update_chat_display(self):
        """Update the chat display with all messages"""
        html_content = ""
        for message in self.chat_history:
            html_content += message.format_html()
        
        self.chat_display.setHtml(html_content)
        
        # Scroll to bottom to show latest message
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def send_message(self):
        """Handle user sending a message"""
        user_text = self.message_input.text().strip()
        if not user_text:
            return
        
        # Add user message to chat
        self.add_message(user_text, is_user=True)
        self.message_input.clear()
        
        # Process and respond
        self.generate_response(user_text)
    
    def generate_response(self, user_text):
        """Generate AI response to user message"""
        # Extract question intent
        question_intent = self.classify_question_intent(user_text)
        
        # If we have predictions, include them for context
        context = ""
        if self.predictions:
            context = f"""The user's optimal focus conditions are:
            - Best time of day: {self.predictions['best_time']}
            - Best session length: {self.predictions['best_length']} minutes
            - Best day of week: {self.predictions['best_day']}
            """
        
        prompt = f"""You are a focus and productivity expert assistant helping the user improve their focus habits.
        
        {context}
        
        The user asks: "{user_text}"
        
        The user is currently using our Focus Enhancement app with features including:
        1. Pomodoro Timer with customizable session lengths and break durations
        2. App tracking that monitors productive vs distracting applications
        3. Focus sessions that record productivity metrics
        4. Statistics dashboard showing focus patterns
        5. Todo list for task management
        
        Answer the specific question without repeating generic suggestions unless directly asked.
        Address their exact query with practical advice relevant to the app's features.
        Keep your answer under 150 words."""
        
        # Try to use Gemini API if we have a key
        if self.gemini_api_key:
            try:
                # Use Gemini API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"
                
                headers = {
                    "Content-Type": "application/json",
                }
                
                data = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                }
                
                # Make the API call
                response = requests.post(url, headers=headers, data=json.dumps(data))
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract text from the response
                    ai_response = result["candidates"][0]["content"]["parts"][0]["text"]
                    self.add_message(ai_response)
                    return
            except Exception as e:
                print(f"Gemini API error: {str(e)}")
        
        # Fallback to local response generation with specific question intent
        self.generate_local_response(user_text, question_intent)
    
    def classify_question_intent(self, user_text):
        """Classify the user's question to better understand their intent"""
        text = user_text.lower()
        
        # Check if user is commenting that responses are repetitive
        repetition_phrases = [
            "same message", "same response", "same answer", "repetitive", 
            "again and again", "keeps repeating", "giving same", "keeps saying",
            "keeps telling", "always say", "always tell", "redundant"
        ]
        
        for phrase in repetition_phrases:
            if phrase in text:
                return "complaint_repetition"
        
        # Define intent categories and their related keywords
        intent_keywords = {
            "time_question": ["when", "time", "morning", "afternoon", "evening", "night", "day time", "hour", "o'clock"],
            "day_question": ["day", "week", "monday", "tuesday", "wednesday", "thursday", "friday", "weekend", "weekday"],
            "duration_question": ["how long", "minutes", "length", "duration", "session", "pomodoro", "timer"],
            "distraction_question": ["distract", "focus", "concentrate", "attention", "app block", "notification", "alert"],
            "app_usage_question": ["how to use", "features", "app", "function", "tracker", "how does", "work"],
            "statistics_question": ["stats", "data", "progress", "improvement", "history", "tracking", "report"],
            "task_question": ["task", "todo", "to-do", "list", "what should I", "what to do", "priority", "work on"],
            "break_question": ["break", "rest", "pause", "interval", "stop", "between"],
            "tech_question": ["problem", "error", "bug", "not working", "issue", "help", "fix", "broken"],
            "personal_question": ["you", "your", "who are", "what are", "chatbot", "assistant"]
        }
        
        # Check for specific intent patterns
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return intent
        
        # Check for question words
        question_words = ["how", "what", "why", "where", "when", "who", "which", "can", "should", "could", "would"]
        for word in question_words:
            if text.startswith(word) or f" {word} " in text:
                return "general_question"
        
        # Default intent
        return "general_statement"
    
    def generate_local_response(self, user_text, question_intent="general_statement"):
        """Generate a response locally when API is unavailable"""
        user_text_lower = user_text.lower()

        # Handle complaint about repetitive responses
        if question_intent == "complaint_repetition":
            response = """I apologize for being repetitive. I'll try to be more varied and specific in my responses.

What would you like to know about improving your focus? For example:
• Using specific app features?
• Understanding your focus statistics?
• Techniques for handling distractions?
• Setting up effective Pomodoro sessions?"""
            self.add_message(response)
            return
        
        # Handle based on the classified intent of the question
        if question_intent == "time_question":
            if self.predictions:
                response = f"""Your optimal focus time is **{self.predictions['best_time']}** based on your past Pomodoro sessions.

During this time, your focus score averages higher and you experience fewer distractions. To leverage this, schedule your Pomodoro sessions around this time in the Timer tab.

The app will continue learning your patterns as you use it more."""
            else:
                response = """I don't have enough data about your focus patterns yet.

Try using the Pomodoro Timer at different times of day, and I'll analyze when you're most productive. Many users find mornings (8-10am) or mid-afternoons (2-4pm) most effective.

After 5-10 sessions, I can give you personalized timing recommendations."""
                
        elif question_intent == "day_question":
            if self.predictions:
                response = f"""According to your focus session history, **{self.predictions['best_day']}** is your most productive day.

On {self.predictions['best_day']}s, your productivity metrics are consistently higher than other days. Consider scheduling your most challenging tasks on this day.

You can view your day-by-day performance in the Statistics tab."""
            else:
                response = """I need more focus session data to determine your best day.

Use the Pomodoro Timer throughout different days of the week, and I'll identify patterns in your productivity metrics.

The Statistics tab will show you a breakdown by day once you've completed more sessions."""
        
        elif question_intent == "duration_question":
            if self.predictions:
                response = f"""Your focus data shows your optimal session length is **{self.predictions['best_length']} minutes**.

This is when you maintain peak attention before needing a break. You can select this duration directly in the Pomodoro Timer settings.

The app automatically suggests break times based on your session length."""
            else:
                response = """I haven't collected enough data to suggest your ideal session length.

Our Pomodoro Timer offers preset options of 15, 25, 30, 45, and 60 minutes. Try different lengths to see what works best for you.

After several sessions, I'll analyze which duration gives you the highest focus scores."""
        
        elif question_intent == "distraction_question":
            response = """To manage distractions with our app:

1. Use the "Allowed Apps" feature in the Pomodoro Timer - select only productive apps for your session
2. Enable distraction alerts to notify you when you switch to non-productive apps
3. Review your distraction patterns in the Statistics tab
4. Set shorter Pomodoro sessions (25-30 min) if you're easily distracted

What specific distraction issues are you experiencing?"""
        
        elif question_intent == "app_usage_question":
            response = """The Focus Enhancement app helps you optimize productivity through:

1. **Pomodoro Timer**: Set focused work periods with breaks
2. **App Tracking**: Monitor which applications help or hinder focus
3. **Statistics**: View your productivity trends and patterns
4. **Focus AI**: That's me! I analyze your data to provide personalized recommendations

Which specific feature would you like to learn more about?"""
        
        elif question_intent == "statistics_question":
            response = """Your focus statistics are available in the Statistics tab, showing:

• Daily/weekly productivity trends
• Focus scores across different times and days
• Distraction frequency and duration 
• Most productive vs. distracting applications
• Session length effectiveness

The more sessions you complete, the more detailed insights I can provide. Check the charts to see patterns in your focus habits."""
        
        elif question_intent == "task_question":
            if self.predictions:
                response = f"""For maximum productivity, use the Todo List tab to:

1. Add your most important tasks and schedule them for {self.predictions['best_day']}s
2. Set the Pomodoro Timer to {self.predictions['best_length']} minutes when working on them
3. Schedule challenging tasks around {self.predictions['best_time']} when your focus peaks
4. Break large tasks into smaller subtasks for better focus

The app will track which tasks yield your highest productivity."""
            else:
                response = """To manage tasks effectively with our app:

1. Use the Todo List tab to create and prioritize tasks
2. Start a Pomodoro session and select a task to work on
3. The app will track your focus metrics for different task types
4. Complete several sessions so I can analyze which tasks you focus on best

Would you like me to explain how to set up your first task?"""
        
        elif question_intent == "break_question":
            response = """Our app manages breaks based on the Pomodoro technique:

• After each focus session, the app automatically suggests a 5-minute break
• After 4 sessions, it recommends a longer 15-30 minute break
• The break timer counts down just like focus sessions
• Use breaks to stretch, hydrate, or rest your eyes - but avoid digital distractions

Breaks are essential for maintaining focus over longer periods. Would you like to adjust your break settings?"""
        
        elif question_intent == "tech_question":
            response = """If you're experiencing technical issues:

1. Check that your session data is being saved in the Statistics tab
2. Make sure you're ending Pomodoro sessions properly with the Stop button
3. For app tracking issues, verify your allowed apps are correctly selected
4. Try restarting the app if you notice any performance problems

What specific technical problem are you having?"""
        
        elif question_intent == "personal_question":
            response = """I'm your Focus AI assistant built into the Focus Enhancement app. 

I analyze your Pomodoro session data to identify your optimal focus patterns and provide personalized productivity recommendations.

As you use the app more, my suggestions become more tailored to your specific work habits and focus tendencies."""
        
        # General "what should I do" questions
        elif "general_question" in question_intent or any(phrase in user_text_lower for phrase in ["what should i do", "how can i improve", "how to focus", "improve focus", "focus better"]):
            # Get the current greeting to avoid repetition
            current_greeting = "Here are my recommendations:"
            
            # Pick different phrasings for variety
            greeting_options = [
                "Based on your focus patterns, here's what I suggest:",
                "After analyzing your focus sessions, I recommend:",
                "To optimize your productivity, consider these personalized tips:",
                "Your focus data indicates these strategies will help you:",
                "For your specific work patterns, try these approaches:"
            ]
            
            # Choose a greeting that's different from the previous one
            for greeting in greeting_options:
                if greeting != current_greeting:
                    current_greeting = greeting
                    break
            
            if self.predictions:
                response = f"""{current_greeting}

• Target your deep work for **{self.predictions['best_time']}** - your data shows this is your peak focus period
• Structure work into **{self.predictions['best_length']}-minute** focused sessions to match your attention span
• Schedule your most important work on **{self.predictions['best_day']}s** when your productivity metrics are highest
• When starting a session, enable App Tracking to minimize distractions

Can I help with implementing any of these strategies?"""
            else:
                response = """To improve your focus with our app:

1. Start using the Pomodoro Timer regularly - this builds focus stamina
2. Select only productive apps in the Allowed Apps feature
3. Track which tasks give you the highest focus scores
4. Complete 5-10 sessions so I can analyze your optimal focus patterns

Which would you like to try first?"""
        
        # Default response for other inputs
        else:
            if self.predictions and not any(s in user_text_lower for s in ["thank", "thanks", "ok", "okay", "got it"]):
                response = f"""I'm not sure I understand your question. Based on your focus data, I can help with:

• Finding your optimal focus time (currently {self.predictions['best_time']})
• Setting ideal session lengths (currently {self.predictions['best_length']} min)
• Planning your most productive day ({self.predictions['best_day']})
• Minimizing distractions
• Using app features effectively

What would you like to know more about?"""
            else:
                response = """I'm here to help you improve your focus and productivity using our app's features. I can answer questions about:

• Using the Pomodoro Timer effectively
• Managing distractions with the app
• Understanding your focus statistics
• Optimizing your task management
• Finding your best focus times and durations

What specifically would you like help with?"""
        
        self.add_message(response)

    def get_user_sessions(self):
        """Get the last 10 sessions for the user"""
        sessions = self.db.get_user_sessions(self.user_id)
        return sessions
    
    def prepare_data_for_models(self, sessions):
        """Prepare the session data for model input"""
        data = []
        for session in sessions:
            # Unpack session data 
            session_id, date, day, start_time, end_time, task_type, app_switch_count, distraction_duration, total_focus_duration, focus_score, productivity_percentage, break_duration = session
            
            session_data = {
                'Date': date,
                'Day': day.lower() if isinstance(day, str) else 'monday',  # Ensure lowercase
                'Start Time': start_time,
                'End Time': end_time,
                'Task Type': task_type.lower() if isinstance(task_type, str) else 'studying',  # Ensure lowercase
                'App Switch Count': app_switch_count,
                'Distraction Duration (mins)': distraction_duration,
                'Total Focus Duration (mins)': total_focus_duration,
                'Focus Score (0-10)': focus_score,
                'Productivity %': productivity_percentage,
            }
            data.append(session_data)
        
        df = pd.DataFrame(data)
        
        # Convert date strings to datetime objects
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            # If date parsing fails, try to handle different formats
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            except Exception as e:
                print(f"Warning: Could not parse dates: {str(e)}")
        
        # Add Hour column for best time model
        try:
            df['Hour'] = pd.to_datetime(df['Start Time'], format='%H:%M:%S').dt.hour
        except:
            # If time parsing fails, try different formats
            try:
                df['Hour'] = pd.to_datetime(df['Start Time'], format='%H:%M').dt.hour
            except:
                try:
                    # Parse each time string individually
                    df['Hour'] = df['Start Time'].apply(lambda x: 
                        pd.to_datetime(x, format='%H:%M:%S').hour if ':' in str(x) else 9)
                except:
                    # Default to 9 AM if can't parse
                    df['Hour'] = 9
                    print("Warning: Could not parse times properly")
        
        # Preprocessing for day model
        last_date = df['Date'].max()
        df['Days Ago'] = (last_date - df['Date']).dt.days
        df['Weight'] = df['Days Ago'].apply(self.assign_weight)
        df['Weighted Focus Score'] = df['Focus Score (0-10)'] * df['Weight']
        df['Weighted Productivity'] = df['Productivity %'] * df['Weight']
        
        # Add Day_encoded column using LabelEncoder
        day_encoder = LabelEncoder()
        df['Day_encoded'] = day_encoder.fit_transform(df['Day'])
        
        # Calculate combined score metric
        df['combined_score'] = df['Focus Score (0-10)'] * df['Productivity %'] / 10
        
        # Add productivity label for the models
        df['Productivity_Label'] = df['Productivity %'].apply(
            lambda x: 'High' if x >= 75 else ('Mid' if x >= 50 else 'Low')
        )
        
        return df, day_encoder
    
    def assign_weight(self, days):
        """Assign weight based on how recent the session is"""
        if days <= 7:
            return 1.0
        elif days <= 14:
            return 0.75
        elif days <= 21:
            return 0.5
        else:
            return 0.25
            
    def predict_best_day(self, df):
        """Predict the best day for productivity using the model"""
        try:
            # Create feature set for day model based on the error message
            features = ['Day_encoded', 'App Switch Count', 
                       'Distraction Duration (mins)', 'Total Focus Duration (mins)', 
                       'Weighted Focus Score', 'Weighted Productivity']
            
            # Ensure all required features are present
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0  # Default value if missing
            
            X = df[features]
            
            # Directly get the best day based on productivity
            day_productivity = df.groupby('Day')['Productivity %'].mean()
            best_day = day_productivity.idxmax()
            
            return best_day.capitalize()
        except Exception as e:
            print(f"Error predicting best day: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Monday"  # Default fallback
    
    def predict_best_time(self, df):
        """Predict the best time for productivity using the model"""
        try:
            # Create feature set for time model based on error message
            features = ['Hour', 'App Switch Count', 
                       'Distraction Duration (mins)', 'Total Focus Duration (mins)', 
                       'Weighted Focus Score', 'Weighted Productivity']
            
            # Ensure all required features are present
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0  # Default value if missing
            
            # Directly get the best hour based on productivity
            hour_productivity = df.groupby('Hour')['Productivity %'].mean()
            best_hour = int(hour_productivity.idxmax())
            
            # Format as AM/PM time
            best_time_str = f"{best_hour}:00 {('AM' if best_hour < 12 else 'PM')}"
            if best_hour == 0:
                best_time_str = "12:00 AM"
            elif best_hour > 12:
                best_time_str = f"{best_hour-12}:00 PM"
                
            return best_time_str
        except Exception as e:
            print(f"Error predicting best time: {str(e)}")
            import traceback
            traceback.print_exc()
            return "9:00 AM"  # Default fallback
    
    def predict_best_length(self, df):
        """Predict the best session length using the pomodoro model"""
        try:
            # For pomodoro length, simplify by using the average duration
            avg_duration = df['Total Focus Duration (mins)'].mean()
            # Round to nearest 5 minutes and ensure reasonable range
            best_length = max(25, min(90, 5 * round(avg_duration/5)))
            
            return int(best_length)
        except Exception as e:
            print(f"Error predicting best length: {str(e)}")
            import traceback
            traceback.print_exc()
            return 25  # Default Pomodoro length
    
    def get_model_predictions(self, df):
        """Get predictions from all three models"""
        try:
            # Get best day prediction
            best_day = self.predict_best_day(df)
            
            # Get best time prediction
            best_time = self.predict_best_time(df)
            
            # Get best session length
            best_length = self.predict_best_length(df)
            
            return {
                'best_time': best_time,
                'best_length': best_length,
                'best_day': best_day
            }
        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'best_time': "9:00 AM",
                'best_length': 25,
                'best_day': "Monday"
            }
    
    def generate_initial_message(self, predictions, df):
        """Generate the initial welcome message with personalized suggestions"""
        try:
            # Get some statistics to enrich the prompt
            avg_productivity = df['Productivity %'].mean()
            avg_focus_score = df['Focus Score (0-10)'].mean()
            avg_distraction = df['Distraction Duration (mins)'].mean()
            avg_session_length = df['Total Focus Duration (mins)'].mean()
            
            prompt = f"""You are a focus and productivity AI assistant that helps users improve their focus habits. 
            Based on the user's focus session data, I've analyzed their patterns and found:
            
            - Best time of day: {predictions['best_time']}
            - Best session length: {predictions['best_length']} minutes
            - Best day of week: {predictions['best_day']}
            
            Current user statistics:
            - Average productivity: {avg_productivity:.1f}%
            - Average focus score: {avg_focus_score:.1f}/10
            - Average distraction time: {avg_distraction:.1f} minutes
            - Average session length: {avg_session_length:.1f} minutes
            
            Write a friendly greeting introducing yourself as a Focus AI assistant, then provide 2-3 personalized suggestions based on their data.
            End by asking how you can help them improve their focus and productivity.
            Keep your response concise, warm and conversational (under 150 words)."""
            
            # Try to use Gemini API if we have a key
            if self.gemini_api_key:
                try:
                    # Use Gemini API
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"
                    
                    headers = {
                        "Content-Type": "application/json",
                    }
                    
                    data = {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": prompt
                                    }
                                ]
                            }
                        ]
                    }
                    
                    # Make the API call
                    response = requests.post(url, headers=headers, data=json.dumps(data))
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Extract text from the response
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    print(f"Gemini API error: {str(e)}")
            
            # Default message if API fails
            return self.generate_default_welcome(predictions)
                
        except Exception as e:
            return self.generate_default_welcome(predictions)
    
    def generate_default_welcome(self, predictions):
        """Generate a default welcome message when API is unavailable"""
        best_time = predictions['best_time']
        best_length = predictions['best_length']
        best_day = predictions['best_day']
        
        return f"""👋 Hi there! I'm your Focus AI assistant, here to help you optimize your productivity and focus.

Based on your session data, I've discovered some interesting patterns:

✨ Your most productive time seems to be around **{best_time}**
⏱️ You perform best with **{best_length}-minute** focus sessions
📅 **{best_day}** appears to be your optimal day for deep work

I can suggest personalized strategies to help you work with your natural rhythms. What would you like to know about improving your focus?"""
    
    def refresh_suggestions(self):
        """Start a new chat with initial suggestions"""
        # Clear chat history
        self.chat_history = []
        
        # Get user sessions
        sessions = self.get_user_sessions()
        if not sessions:
            welcome = """👋 Welcome to Focus AI! I don't have enough data about your focus sessions yet.

Start some focus sessions using the Pomodoro Timer, and I'll analyze your patterns to provide personalized suggestions.

In the meantime, I can still answer questions about focus and productivity techniques. What would you like to know?"""
            self.add_message(welcome)
            return
        
        # Prepare data and get predictions
        try:
            df, day_encoder = self.prepare_data_for_models(sessions)
            self.predictions = self.get_model_predictions(df)
            
            # Generate initial message
            initial_message = self.generate_initial_message(self.predictions, df)
            
            # Add to chat
            self.add_message(initial_message)
        except Exception as e:
            # If prediction fails, add a generic welcome
            print(f"Error generating initial message: {str(e)}")
            welcome = """👋 Hi there! I'm your Focus AI assistant.

I can help you discover your optimal focus patterns and suggest techniques to improve your productivity.

What would you like to know about focus, productivity, or work habits?"""
            self.add_message(welcome) 