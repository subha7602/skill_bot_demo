# reinforcement_learning.py
import json
import os
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReinforcementLearningManager:
    """
    Manages the reinforcement learning process using user feedback to improve responses.
    Uses AWS Bedrock with Claude Haiku by default.
    """
    
    def __init__(self, model_id="anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Initialize the reinforcement learning manager.
        
        Args:
            model_id: The Bedrock model ID to use (defaults to Claude Haiku)
        """
        self.model_id = model_id
        self.bedrock_runtime = self._initialize_bedrock_client()
    
    def _initialize_bedrock_client(self):
        """Initialize the AWS Bedrock client."""
        try:
            return boto3.client('bedrock-runtime')
        except Exception as e:
            logger.error(f"Error initializing Bedrock client: {e}")
            # Return None but don't fail - we'll handle missing client in the methods
            return None
    
    def get_enhanced_response(self, user_query):
        """
        Generate a response that incorporates learnings from past feedback.
        
        Args:
            user_query: The user's question
            
        Returns:
            An enhanced response
        """
        if not self.bedrock_runtime:
            logger.warning("Bedrock client not available. Using fallback method.")
            return self._get_fallback_response(user_query)
        
        # Get feedback handler from Streamlit session state
        import streamlit as st
        if not hasattr(st.session_state, "feedback_handler"):
            logger.warning("Feedback handler not available. Using fallback method.")
            return self._get_fallback_response(user_query)
        
        feedback_handler = st.session_state.feedback_handler
        
        # Get positive and negative examples
        positive_examples = feedback_handler.get_positive_examples(limit=3)
        negative_examples = feedback_handler.get_negative_examples(limit=2)
        
        # Format examples for the prompt
        pos_formatted = ""
        for i, ex in enumerate(positive_examples, 1):
            pos_formatted += f"Example {i}:\nQuery: {ex.get('query', '')}\nGood Response: {ex.get('response', '')}\n\n"
        
        neg_formatted = ""
        for i, ex in enumerate(negative_examples, 1):
            neg_formatted += f"Example {i}:\nQuery: {ex.get('query', '')}\nResponse to Improve: {ex.get('response', '')}\n\n"
        
        # Create enhanced system content for Claude
        system_content = """
        You are a helpful AI assistant specialized in answering questions about employee resumes and skills.
        Provide concise, thorough, and contextual answers to questions about employee skills, experiences, and qualifications.
        Focus on relevant information rather than just listing facts.
        Suggest specific employees for projects based on their skills when appropriate.
        """
        
        # Create user content that includes examples
        user_content = f"""
        I want you to improve your responses based on past feedback.
        
        # Positive Examples (responses users liked):
        {pos_formatted}
        
        # Examples to Improve Upon (responses users disliked):
        {neg_formatted}
        
        Guidelines based on feedback:
        1. Be concise but thorough
        2. Focus on relevant skills from resumes
        3. Provide specific examples when possible
        4. Suggest relevant employees for projects based on skills
        5. Contextualize information rather than just listing facts
        
        Now, please answer this question about employee resumes:
        {user_query}
        """
        
        try:
            # Call Bedrock with the enhanced prompt using Claude model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.5,
                    "system": system_content,
                    "messages": [
                        {
                            "role": "user", 
                            "content": user_content
                        }
                    ]
                })
            )
            
            # Parse the response
            response_body = json.loads(response.get('body').read())
            return response_body.get('content')[0].get('text', '')
            
        except Exception as e:
            logger.error(f"Error calling Bedrock: {e}")
            return self._get_fallback_response(user_query)
    
    def _get_fallback_response(self, user_query):
        """
        Fallback method when Bedrock or feedback handler is not available.
        
        Args:
            user_query: The user's question
            
        Returns:
            A response using the standard query method
        """
        from backend.llm_integration import ContextualProjectRetrieval
        return ContextualProjectRetrieval().query_resumes_improved(user_query)
    
    def export_for_fine_tuning(self, export_path="data/fine_tuning_dataset.jsonl"):
        """
        Export the feedback data in a format suitable for fine-tuning a model.
        
        Args:
            export_path: Path to save the fine-tuning dataset
            
        Returns:
            Path to the exported file
        """
        import streamlit as st
        if not hasattr(st.session_state, "feedback_handler"):
            return None
        
        feedback_handler = st.session_state.feedback_handler
        
        # Get positive examples as they represent good interactions
        positive_examples = feedback_handler.get_positive_examples(limit=1000)
        
        # Convert to the format expected for fine-tuning
        fine_tuning_data = []
        
        for ex in positive_examples:
            entry = {
                "messages": [
                    {"role": "user", "content": ex.get('query', '')},
                    {"role": "assistant", "content": ex.get('response', '')}
                ]
            }
            fine_tuning_data.append(entry)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Write as JSONL (each line is a valid JSON object)
        with open(export_path, 'w') as f:
            for entry in fine_tuning_data:
                f.write(json.dumps(entry) + '\n')
        
        return export_path