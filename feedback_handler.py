# feedback_handler.py
import os
import json
from datetime import datetime
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackHandler:
    """
    Handles the collection, storage, and processing of user feedback for reinforcement learning.
    """
    
    def __init__(self, feedback_file="data/feedback_data.json"):
        """
        Initialize the feedback handler with the path to store feedback.
        
        Args:
            feedback_file: Path to the JSON file where feedback will be stored
        """
        self.feedback_file = feedback_file
        self._ensure_directory_exists()
        self.feedback_data = self._load_feedback_data()
        logger.info(f"FeedbackHandler initialized. Found {len(self.feedback_data)} existing feedback entries.")
    
    def _ensure_directory_exists(self):
        """Ensure the directory for the feedback file exists."""
        try:
            directory = os.path.dirname(self.feedback_file)
            if directory:  # Only try to create if there's a directory component
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
    
    def _load_feedback_data(self):
        """
        Load existing feedback data from file or initialize if it doesn't exist.
        
        Returns:
            List of feedback items
        """
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded feedback data from {self.feedback_file}")
                return data
        except FileNotFoundError:
            logger.warning(f"Feedback file not found at {self.feedback_file}. Creating new file.")
            return []
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in feedback file. Creating new file.")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading feedback data: {e}")
            return []
    
    def _save_feedback_data_safe(self):
        """Save the current feedback data to file using a safe approach that won't corrupt the file."""
        try:
            # Create a temporary file first to avoid corruption
            temp_file = f"{self.feedback_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            
            # If successful, rename to the actual file
            import os
            if os.path.exists(self.feedback_file):
                os.replace(temp_file, self.feedback_file)
            else:
                os.rename(temp_file, self.feedback_file)
                
            logger.info(f"Successfully saved {len(self.feedback_data)} feedback items to {self.feedback_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
            return False
    
    def record_feedback(self, feedback_item):
        """
        Record a new feedback item immediately without disrupting UI.
        Uses thread-safe approaches to prevent issues.
        
        Args:
            feedback_item: Dictionary containing feedback information
                {
                    "id": unique_id,
                    "query": user_query,
                    "response": ai_response,
                    "feedback": "positive" or "negative",
                    "timestamp": ISO format timestamp
                }
                
        Returns:
            Boolean indicating success
        """
        try:
            # Check if this feedback ID already exists
            existing_index = None
            for i, item in enumerate(self.feedback_data):
                if item.get("id") == feedback_item.get("id"):
                    existing_index = i
                    break
            
            # Add metadata
            feedback_item["recorded_at"] = datetime.now().isoformat()
            
            # Update existing or add new
            if existing_index is not None:
                self.feedback_data[existing_index] = feedback_item
                logger.info(f"Updated existing feedback for ID: {feedback_item.get('id')}")
            else:
                self.feedback_data.append(feedback_item)
                logger.info(f"Added new feedback with ID: {feedback_item.get('id')}")
            
            # Save to disk using a background approach that won't disrupt UI
            import threading
            save_thread = threading.Thread(target=self._save_feedback_data_safe)
            save_thread.daemon = True  # Background thread that won't block app exit
            save_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    def get_feedback_stats(self):
        """
        Get statistics about the collected feedback.
        
        Returns:
            Dictionary with feedback statistics
        """
        try:
            if not self.feedback_data:
                return {"total": 0, "positive": 0, "negative": 0, "positive_ratio": 0}
            
            total = len(self.feedback_data)
            positive = sum(1 for item in self.feedback_data if item.get("feedback") == "positive")
            negative = sum(1 for item in self.feedback_data if item.get("feedback") == "negative")
            
            return {
                "total": total,
                "positive": positive,
                "negative": negative,
                "positive_ratio": positive / total if total > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {"total": 0, "positive": 0, "negative": 0, "positive_ratio": 0, "error": str(e)}
    
    def export_feedback_dataset(self, format="json"):
        """
        Export the feedback data in the specified format for training.
        
        Args:
            format: The format to export (json, csv)
            
        Returns:
            Path to the exported file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure directory exists
            export_dir = "data"
            os.makedirs(export_dir, exist_ok=True)
            
            if format.lower() == "csv":
                export_path = f"{export_dir}/feedback_export_{timestamp}.csv"
                df = pd.DataFrame(self.feedback_data)
                df.to_csv(export_path, index=False)
            else:  # default to json
                export_path = f"{export_dir}/feedback_export_{timestamp}.json"
                with open(export_path, 'w') as f:
                    json.dump(self.feedback_data, f, indent=2)
            
            logger.info(f"Exported feedback data to {export_path}")
            return export_path
        except Exception as e:
            logger.error(f"Error exporting feedback dataset: {e}")
            return None
    
    def get_positive_examples(self, limit=100):
        """
        Get examples with positive feedback for fine-tuning.
        
        Args:
            limit: Maximum number of examples to return
            
        Returns:
            List of positive feedback examples
        """
        try:
            positive_examples = [
                item for item in self.feedback_data 
                if item.get("feedback") == "positive"
            ][:limit]
            
            return positive_examples
        except Exception as e:
            logger.error(f"Error getting positive examples: {e}")
            return []
    
    def get_negative_examples(self, limit=100):
        """
        Get examples with negative feedback for improvement analysis.
        
        Args:
            limit: Maximum number of examples to return
            
        Returns:
            List of negative feedback examples
        """
        try:
            negative_examples = [
                item for item in self.feedback_data 
                if item.get("feedback") == "negative"
            ][:limit]
            
            return negative_examples
        except Exception as e:
            logger.error(f"Error getting negative examples: {e}")
            return []