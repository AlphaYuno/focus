from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QListWidget, QListWidgetItem,
                             QMessageBox, QFrame, QTextEdit, QDialog, QFormLayout,
                             QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

class TaskDialog(QDialog):
    """Dialog for adding or editing a task."""
    
    def __init__(self, parent=None, task_title="", task_description=""):
        super().__init__(parent)
        self.setWindowTitle("Task Details")
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QFormLayout()
        
        # Task title
        self.title_input = QLineEdit(task_title)
        self.title_input.setPlaceholderText("Enter task title")
        layout.addRow("Title:", self.title_input)
        
        # Task description
        self.description_input = QTextEdit(task_description)
        self.description_input.setPlaceholderText("Enter task description (optional)")
        layout.addRow("Description:", self.description_input)
        
        # Task type
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["Coding", "Writing", "Studying", "Reading", "Meeting", "Other"])
        layout.addRow("Task Type:", self.task_type_combo)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addRow("", button_layout)
        
        self.setLayout(layout)
    
    def get_task_data(self):
        """Return the task data entered by the user."""
        return {
            "title": self.title_input.text().strip(),
            "description": self.description_input.toPlainText().strip(),
            "task_type": self.task_type_combo.currentText()
        }


class TodoWidget(QWidget):
    def __init__(self, db, user_id):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.init_ui()
        self.load_tasks()
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("To-Do List")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Add task button
        add_task_button = QPushButton("Add Task")
        add_task_button.clicked.connect(self.add_task)
        main_layout.addWidget(add_task_button)
        
        # Task list
        self.task_list = QListWidget()
        self.task_list.setAlternatingRowColors(True)
        self.task_list.itemClicked.connect(self.on_task_clicked)
        main_layout.addWidget(self.task_list)
        
        # Task actions
        actions_layout = QHBoxLayout()
        
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_task)
        self.edit_button.setEnabled(False)
        
        self.complete_button = QPushButton("Complete")
        self.complete_button.clicked.connect(self.complete_task)
        self.complete_button.setEnabled(False)
        
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_task)
        self.delete_button.setEnabled(False)
        
        actions_layout.addWidget(self.edit_button)
        actions_layout.addWidget(self.complete_button)
        actions_layout.addWidget(self.delete_button)
        
        main_layout.addLayout(actions_layout)
        
        # Set the layout
        self.setLayout(main_layout)
        
    def load_tasks(self):
        """Load active tasks from the database."""
        self.task_list.clear()
        tasks = self.db.get_tasks(self.user_id, status="active")
        
        for task_id, title, description, created_at in tasks:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, task_id)
            item.setData(Qt.UserRole + 1, description)
            self.task_list.addItem(item)
    
    def add_task(self):
        """Open dialog to add a new task."""
        dialog = TaskDialog(self)
        if dialog.exec_():
            task_data = dialog.get_task_data()
            
            if not task_data["title"]:
                QMessageBox.warning(self, "Error", "Task title cannot be empty.")
                return
            
            success, message, task_id = self.db.add_task(
                self.user_id, 
                task_data["title"], 
                task_data["description"]
            )
            
            if success:
                self.load_tasks()
            else:
                QMessageBox.warning(self, "Error", message)
    
    def on_task_clicked(self, item):
        """Handle task selection."""
        self.edit_button.setEnabled(True)
        self.complete_button.setEnabled(True)
        self.delete_button.setEnabled(True)
    
    def complete_task(self):
        """Mark the selected task as completed."""
        current_item = self.task_list.currentItem()
        if not current_item:
            return
        
        task_id = current_item.data(Qt.UserRole)
        success, message = self.db.update_task_status(task_id, "completed")
        
        if success:
            self.load_tasks()
            self.edit_button.setEnabled(False)
            self.complete_button.setEnabled(False)
            self.delete_button.setEnabled(False)
        else:
            QMessageBox.warning(self, "Error", message)
    
    def delete_task(self):
        """Delete the selected task."""
        current_item = self.task_list.currentItem()
        if not current_item:
            return
        
        task_id = current_item.data(Qt.UserRole)
        
        reply = QMessageBox.question(
            self, 
            "Confirm Deletion", 
            "Are you sure you want to delete this task?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success, message = self.db.update_task_status(task_id, "deleted")
            
            if success:
                self.load_tasks()
                self.edit_button.setEnabled(False)
                self.complete_button.setEnabled(False)
                self.delete_button.setEnabled(False)
            else:
                QMessageBox.warning(self, "Error", message)

    def edit_task(self):
        """Open dialog to edit the selected task."""
        current_item = self.task_list.currentItem()
        if not current_item:
            return
        
        task_id = current_item.data(Qt.UserRole)
        title = current_item.text()
        description = current_item.data(Qt.UserRole + 1)
        
        dialog = TaskDialog(self, title, description)
        if dialog.exec_():
            task_data = dialog.get_task_data()
            
            if not task_data["title"]:
                QMessageBox.warning(self, "Error", "Task title cannot be empty.")
                return
            
            success, message = self.db.update_task_details(
                task_id,
                task_data["title"],
                task_data["description"]
            )
            
            if success:
                self.load_tasks()
            else:
                QMessageBox.warning(self, "Error", message) 