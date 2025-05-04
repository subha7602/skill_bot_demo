import streamlit as st
from frontend.side_bar import render_sidebar, load_chat_history_from_json as load_chat_history
from frontend.chat_interface import render_chat_interface
from feedback_handler import FeedbackHandler
from reinforcement_learning import ReinforcementLearningManager
import uuid

st.set_page_config(page_title="Skills Bot", layout="wide")

def main():
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "all_chat_histories" not in st.session_state:
        st.session_state.all_chat_histories = load_chat_history()
    
    # Handle new chat creation
    if st.session_state.get("start_new_chat", False):
        # Create a new chat when explicitly requested via button
        st.session_state.selected_chat_id = str(uuid.uuid4())
        st.session_state.all_chat_histories[st.session_state.selected_chat_id] = []
        st.session_state.chat_history = []
        # Reset the flag after creating a new chat
        st.session_state.start_new_chat = False
    elif "selected_chat_id" not in st.session_state:
        # If no chat is selected (first load), create a new one
        st.session_state.selected_chat_id = str(uuid.uuid4())
        st.session_state.all_chat_histories[st.session_state.selected_chat_id] = []
        st.session_state.chat_history = []
            
    # Initialize feedback-related session state variables
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = {}
        
    if "use_enhanced_model" not in st.session_state:
        st.session_state.use_enhanced_model = False
        
    # Initialize feedback handler and RL manager only once to prevent losing state
    if "feedback_handler" not in st.session_state:
        st.session_state.feedback_handler = FeedbackHandler()
                
        # Load existing feedback into session state for UI display
        for item in st.session_state.feedback_handler.feedback_data:
            if "id" in item:
                st.session_state.feedback_data[item["id"]] = item
        
    # Initialize RL manager with Claude Haiku model only
    if "rl_manager" not in st.session_state:
        st.session_state.rl_manager = ReinforcementLearningManager(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    # Sidebar with original content and settings
    with st.sidebar:
        # Original sidebar content
        render_sidebar()
                

                
        # Feedback settings section
        with st.expander("Feedback Settings", expanded=False):
            # Toggle for enhanced model
            st.session_state.use_enhanced_model = st.toggle(
                "Use feedback-enhanced responses", 
                value=st.session_state.use_enhanced_model,
                help="When enabled, responses will incorporate learnings from past feedback"
            )
                        
            # Feedback statistics
            st.subheader("Feedback Statistics")
            stats = st.session_state.feedback_handler.get_feedback_stats()
                        
            # Display metrics in a compact format for sidebar
            st.metric("Total Feedback", stats["total"])
                        
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("👍 Positive", stats["positive"])
            with metric_cols[1]:
                st.metric("👎 Negative", stats["negative"])
                        
            positive_ratio = stats["positive_ratio"] * 100 if stats["total"] > 0 else 0
            st.metric("Satisfaction Rate", f"{positive_ratio:.1f}%")
                        
            # Export buttons
            st.subheader("Export Options")
                        
            if st.button("Export Feedback Data", help="Export to CSV"):
                export_path = st.session_state.feedback_handler.export_feedback_dataset(format="csv")
                if export_path:
                    st.success(f"Data exported to {export_path}")
                else:
                    st.error("Failed to export")
                        
            if st.button("Export for Fine-tuning", help="Export for LLM fine-tuning"):
                export_path = st.session_state.rl_manager.export_for_fine_tuning()
                if export_path:
                    st.success(f"Data exported to {export_path}")
                else:
                    st.error("Failed to export")

    # Main content - just the chat interface
    st.title("Skills Bot")
    render_chat_interface()

if __name__ == "__main__":
    main()