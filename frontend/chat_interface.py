# chat_interface.py
from backend.llm_integration import ContextualProjectRetrieval
import streamlit as st
import pyperclip
import uuid
import json
import os
from datetime import datetime

# Function to load chat history from JSON file
def load_chat_history_from_json():
    try:
        # Check if the file exists
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r") as f:
                loaded_data = json.load(f)
                
                # Convert the loaded data to the format expected by the application
                all_chats = {}
                for chat_id, chat_data in loaded_data.items():
                    formatted_chat = []
                    for entry in chat_data:
                        # Ensure each entry is in the expected format: [("You", msg), ("AI", response), ("id", qa_id)]
                        if isinstance(entry, list) and len(entry) >= 2:
                            formatted_entry = []
                            user_msg = next((item[1] for item in entry if item[0] == "You"), None)
                            ai_msg = next((item[1] for item in entry if item[0] == "AI"), None)
                            qa_id = next((item[1] for item in entry if item[0] == "id"), str(uuid.uuid4()))
                            
                            if user_msg and ai_msg:
                                formatted_entry = [("You", user_msg), ("AI", ai_msg), ("id", qa_id)]
                                formatted_chat.append(formatted_entry)
                    
                    if formatted_chat:  # Only add if there's actual chat content
                        all_chats[chat_id] = formatted_chat
                
                return all_chats
        return {}
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        return {}

# Initialize session state variables at the very beginning
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "all_chat_histories" not in st.session_state:
    # Load existing chat histories from JSON file
    loaded_chats = load_chat_history_from_json()
    st.session_state.all_chat_histories = loaded_chats if loaded_chats else {}
    
# Function to save chat history to JSON file
def save_chat_history_to_json():
    try:
        with open("chat_history.json", "w") as f:
            json.dump(st.session_state.all_chat_histories, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

# Define callback functions that don't mess with chat state
def handle_feedback(qa_id, user_query, ai_response, is_positive):
    """Unified feedback handler that preserves chat state"""
    # Create feedback item
    feedback_item = {
        "id": qa_id,
        "query": user_query,
        "response": ai_response,
        "feedback": "positive" if is_positive else "negative",
        "timestamp": datetime.now().isoformat(),
        "enhanced_model": st.session_state.get("use_enhanced_model", False)
    }
    
    # Update session state with feedback
    st.session_state.feedback_data[qa_id] = feedback_item
    st.session_state[f"thumbs_up_{qa_id}"] = is_positive
    st.session_state[f"thumbs_down_{qa_id}"] = not is_positive
    
    # Save to persistent storage if feedback handler exists
    if hasattr(st.session_state, "feedback_handler"):
        st.session_state.feedback_handler.record_feedback(feedback_item)

def render_chat_interface():
    """
    Renders the chat interface with safer session state handling
    """
    # Handle new chat creation first - highest priority
    if st.session_state.get("start_new_chat", False):
        st.session_state.selected_chat_id = str(uuid.uuid4())
        st.session_state.all_chat_histories[st.session_state.selected_chat_id] = []
        st.session_state.chat_history = []
        st.session_state.start_new_chat = False
    elif "selected_chat_id" not in st.session_state:
        if st.session_state.all_chat_histories:
            st.session_state.selected_chat_id = list(st.session_state.all_chat_histories.keys())[0]
        else:
            st.session_state.selected_chat_id = str(uuid.uuid4())
            st.session_state.all_chat_histories[st.session_state.selected_chat_id] = []
    elif st.session_state.selected_chat_id not in st.session_state.all_chat_histories:
        # If selected chat ID is invalid, create a new one
        st.session_state.selected_chat_id = str(uuid.uuid4())
        st.session_state.all_chat_histories[st.session_state.selected_chat_id] = []

    # Set chat_history to the currently selected chat - always happens
    st.session_state.chat_history = st.session_state.all_chat_histories[st.session_state.selected_chat_id]
    
    # Chat header
    chat_data = st.session_state.chat_history
    if chat_data and len(chat_data) > 0:
        first_entry = chat_data[0]
        first_question = next((item[1] for item in first_entry if item[0] == "You"), "New Chat")
        st.header(f"💬 Chat: {first_question[:50]}...")
    else:
        st.header("💬 New Chat")
    
    # User input
    user_query = st.chat_input("Ask about employee resumes...")
    
    if user_query:
        # Generate a unique ID for this Q&A pair
        qa_id = str(uuid.uuid4())
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.get("use_enhanced_model", False) and hasattr(st.session_state, "rl_manager"):
                    response = st.session_state.rl_manager.get_enhanced_response(user_query)
                else:
                    response = ContextualProjectRetrieval().query_resumes_improved(user_query)
            
            # Display response
            st.write(response)
            
            # Check for existing feedback
            has_positive = st.session_state.get(f"thumbs_up_{qa_id}", False)
            has_negative = st.session_state.get(f"thumbs_down_{qa_id}", False)
            
            # Show feedback status if present
            if has_positive:
                st.success("👍 You found this helpful")
            elif has_negative:
                st.info("👎 You found this not helpful")
            
            # Button row
            cols = st.columns([0.6, 0.2, 0.2])
            
            # Copy button
            with cols[0]:
                if st.button("📋 Copy", key=f"copy_{qa_id}"):
                    pyperclip.copy(response)
                    st.toast("Copied!")
            
            # Thumbs up button
            with cols[1]:
                disabled = has_positive or has_negative
                button_text = "👍 ✓" if has_positive else "👍"
                
                if st.button(button_text, key=f"thumbs_up_btn_{qa_id}", disabled=disabled):
                    # Store existing chat state in a variable that will not be lost
                    st.session_state[f"_prev_chat_{qa_id}"] = st.session_state.chat_history.copy()
                    
                    # Record feedback without touching chat history
                    handle_feedback(qa_id, user_query, response, True)
                    st.toast("Thank you for your positive feedback!")
                    
                    # CRITICAL: Ensure chat history isn't lost by explicitly restoring it
                    if f"_prev_chat_{qa_id}" in st.session_state:
                        st.session_state.chat_history = st.session_state[f"_prev_chat_{qa_id}"]
            
            # Thumbs down button
            with cols[2]:
                disabled = has_positive or has_negative
                button_text = "👎 ✓" if has_negative else "👎"
                
                if st.button(button_text, key=f"thumbs_down_btn_{qa_id}", disabled=disabled):
                    # Store existing chat state
                    st.session_state[f"_prev_chat_{qa_id}"] = st.session_state.chat_history.copy()
                    
                    # Record feedback
                    handle_feedback(qa_id, user_query, response, False)
                    st.toast("Thank you for your feedback!")
                    
                    # Ensure chat history isn't lost
                    if f"_prev_chat_{qa_id}" in st.session_state:
                        st.session_state.chat_history = st.session_state[f"_prev_chat_{qa_id}"]
        
        # Store conversation in chat history AFTER user interaction
        chat_entry = [("You", user_query), ("AI", response), ("id", qa_id)]
        
        # Safely store in chat history - find and update if exists, otherwise add
        found = False
        for i, entry in enumerate(st.session_state.chat_history):
            if len(entry) > 2 and entry[2][0] == "id" and entry[2][1] == qa_id:
                st.session_state.chat_history[i] = chat_entry
                found = True
                break
                
        if not found:
            st.session_state.chat_history.insert(0, chat_entry)
        
        # Ensure the all_chat_histories for this ID points to the same list as chat_history
        st.session_state.all_chat_histories[st.session_state.selected_chat_id] = st.session_state.chat_history
        
        # Save chat history to JSON file after each interaction
        save_chat_history_to_json()
    
    # Display previous chat history
    if st.session_state.chat_history:  # Only try to display if there's actually history
        for idx, chat_unit in enumerate(st.session_state.chat_history):
            # Extract messages and ID
            user_msg = next((item[1] for item in chat_unit if item[0] == "You"), "")
            ai_msg = next((item[1] for item in chat_unit if item[0] == "AI"), "")
            qa_id = next((item[1] for item in chat_unit if item[0] == "id"), None)
            
            # Skip if we just displayed it (new message)
            if idx == 0 and user_query and user_msg == user_query:
                continue
                
            # Display user message
            with st.chat_message("user"):
                st.write(user_msg)
            
            # Display AI message with feedback
            with st.chat_message("assistant"):
                st.write(ai_msg)
                
                if qa_id:
                    # Check feedback status
                    has_positive = st.session_state.get(f"thumbs_up_{qa_id}", False)
                    has_negative = st.session_state.get(f"thumbs_down_{qa_id}", False)
                    
                    # Show feedback status
                    if has_positive:
                        st.success("👍 You found this helpful")
                    elif has_negative:
                        st.info("👎 You found this not helpful")
                    
                    # Button row
                    cols = st.columns([0.6, 0.2, 0.2])
                    
                    # Copy button
                    with cols[0]:
                        if st.button("📋 Copy", key=f"copy_hist_{idx}"):
                            pyperclip.copy(ai_msg)
                            st.toast("Copied!")
                    
                    # Show feedback buttons if no feedback yet
                    if not has_positive and not has_negative:
                        # Thumbs up button
                        with cols[1]:
                            if st.button("👍", key=f"thumbs_up_hist_{idx}"):
                                # Save current chat state
                                st.session_state[f"_prev_chat_hist_{idx}"] = st.session_state.chat_history.copy()
                                
                                # Record feedback
                                handle_feedback(qa_id, user_msg, ai_msg, True)
                                st.toast("Thank you for your positive feedback!")
                                
                                # Restore chat state
                                if f"_prev_chat_hist_{idx}" in st.session_state:
                                    st.session_state.chat_history = st.session_state[f"_prev_chat_hist_{idx}"]
                        
                        # Thumbs down button
                        with cols[2]:
                            if st.button("👎", key=f"thumbs_down_hist_{idx}"):
                                # Save current chat state
                                st.session_state[f"_prev_chat_hist_{idx}"] = st.session_state.chat_history.copy()
                                
                                # Record feedback
                                handle_feedback(qa_id, user_msg, ai_msg, False)
                                st.toast("Thank you for your feedback!")
                                
                                # Restore chat state
                                if f"_prev_chat_hist_{idx}" in st.session_state:
                                    st.session_state.chat_history = st.session_state[f"_prev_chat_hist_{idx}"]
                    else:
                        # Show disabled buttons if feedback already given
                        with cols[1]:
                            button_text = "👍 ✓" if has_positive else "👍"
                            st.button(button_text, key=f"thumbs_up_disabled_{idx}", disabled=True)
                        
                        with cols[2]:
                            button_text = "👎 ✓" if has_negative else "👎"
                            st.button(button_text, key=f"thumbs_down_disabled_{idx}", disabled=True)