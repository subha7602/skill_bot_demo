# from backend.new import ContextualRetrieval
from backend.llm_integration import ContextualProjectRetrieval
import streamlit as st
import pyperclip 
from datetime import datetime
import json



CHAT_HISTORY_FILE = "chat_history.json"
def load_chat_history_from_json():
    """Load chat history from a JSON file and validate entries."""
    try:
        with open(CHAT_HISTORY_FILE, "r") as f:
            history = json.load(f)
            valid_history = {}
            for chat_id, chat_data in history.items():
                # Only add chat if it has content
                if chat_data and isinstance(chat_data, list) and len(chat_data) > 0:
                    # Check if the chat entries have the expected format
                    valid_entries = []
                    valid_chat = False
                    
                    for entry in chat_data:
                        # Check if entry contains tuples or lists with key-value pairs
                        if isinstance(entry, list) and len(entry) >= 2:
                            # Look for user message, AI response, and ID
                            user_msg = None
                            ai_msg = None
                            qa_id = None
                            
                            for item in entry:
                                if isinstance(item, list) and len(item) == 2:
                                    if item[0] == "You":
                                        user_msg = item[1]
                                    elif item[0] == "AI":
                                        ai_msg = item[1]
                                    elif item[0] == "id":
                                        qa_id = item[1]
                            
                            if user_msg and ai_msg:
                                valid_entries.append(entry)
                                valid_chat = True
                    
                    if valid_chat:
                        valid_history[chat_id] = valid_entries
            
            return valid_history
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        st.error("Error: Chat history file is corrupted")
        return {} 

def save_chat_history(chat_id, chat_data):
    """Save chat history to a JSON file, with validation."""
    try:
        # Validate the chat data before saving
        if not chat_data or not isinstance(chat_data, list) or len(chat_data) == 0:
            # Don't save empty chats
            return load_chat_history_from_json()  # Return existing history without changes

        
        # Load existing history
        history = load_chat_history_from_json()

        # Add or update the chat data for the given chat_id
        history[chat_id] = chat_data

        # Save back to the JSON file
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)

        # Return the updated history
        return history
    except Exception as e:
        st.error(f"Error saving chat history: {e}")
        return None
def delete_chat_history(chat_id):
    """Delete a specific chat from the chat history file."""
    try:
        # Load existing history
        history = load_chat_history_from_json()
        
        # Remove the chat if it exists
        if chat_id in history:
            del history[chat_id]
            
            # Save back to the JSON file
            with open(CHAT_HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=4)
            
            # Update session state
            st.session_state.all_chat_histories = history
            
            # If the deleted chat was selected, clear the selection
            if st.session_state.selected_chat_id == chat_id:
                st.session_state.selected_chat_id = None
                st.session_state.chat_history = []
            
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting chat history: {e}")
        return False

def clear_chat_history():
    """Clears the current chat history and initializes a new chat."""
    st.session_state.chat_history = []  # Clear the current chat

def load_selected_chat(chat_id):
    """Loads a selected chat history into the current chat session."""
    if chat_id in st.session_state.all_chat_histories:
        st.session_state.chat_history = st.session_state.all_chat_histories[chat_id]
        st.session_state.selected_chat_id = chat_id
        return True
    return False

def render_chat_interface():
    """
    Renders the chat interface for interacting with the resume AI.
    Includes a compact copy button with a subtle toast notification.
    """

    # Chat header with selected chat indicator
    if st.session_state.selected_chat_id:
        selected_chat_data = st.session_state.all_chat_histories.get(st.session_state.selected_chat_id)
        if selected_chat_data and len(selected_chat_data) > 0 and len(selected_chat_data[0]) > 0:
            first_question = selected_chat_data[0][0][1] if selected_chat_data[0][0] else "No questions asked"
        else:
            first_question = "No questions asked"
        st.header(f"💬 Chat: {first_question[:50]}...")
    else:
        st.header("💬 Chat with Skills Bot")

    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input with a send button
    user_query = st.chat_input("Ask about employee resumes...")

    if user_query:
        response = ContextualProjectRetrieval().query_resumes_improved(user_query)
        st.session_state.chat_history.insert(0, [("You", user_query), ("AI", response)])

    # Display chat history
    for idx, chat_unit in enumerate(st.session_state.chat_history):
        for sender, message in chat_unit:
            with st.chat_message("user" if sender == "You" else "assistant"):
                col1, col2 = st.columns([0.85, 0.15])

                with col1:
                    st.write(message)

                # Add copy button only for AI responses
                if sender == "AI":
                    button_key = f"copy_{idx}"

                    with col2:
                        if st.button("📋", key=button_key, help="Copy to clipboard"):
                            pyperclip.copy(message)  # Copy text to clipboard
                            st.toast("Copied!")  # Compact toast notification
def render_sidebar():
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;font-size:30px;'>SKILLS BOT</h1>", unsafe_allow_html=True)

        # Display logo (if available)
        try:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("logo.jpg", width=150)
        except FileNotFoundError:
            st.warning("⚠️ Logo not found! Please add `logo.jpg` to the root folder.")

        st.markdown("<hr style='margin: 0px ;'>", unsafe_allow_html=True)

        # New Chat Button using Columns
        col1, col2 = st.columns([3, 2])  # Adjust column widths as needed
        with col1:
            pass  # Empty column to push the button to the right
        with col2:
            # Apply styling
            st.markdown(
                """
                <style>
                div.new-chat-button > div.stButton > button {
                    border: 1px solid white;
                    margin: 5px;
                    text-align: center;
                    display: inline-block;
                    font-size: 14px;
                    cursor: pointer;
                    border-radius: 5px;
                    width: 100%;
                }
                .chat-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 5px;
                    border-radius: 4px;
                    margin-bottom: 5px;
                    cursor: pointer;
                }
                .chat-row:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                }
                .chat-title {
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    flex-grow: 1;
                }
                .delete-icon {
                    flex-shrink: 0;
                    color: #ff4b4b;
                    visibility: hidden;
                    margin-left: 8px;
                }
                .chat-row:hover .delete-icon {
                    visibility: visible;
                }
                .confirm-dialog {
                    background-color: rgba(255, 255, 255, 0.1);
                    padding: 10px;
                    border-radius: 4px;
                    margin-bottom: 10px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            with st.container():
                st.markdown('<div class="new-chat-button">', unsafe_allow_html=True)
                new_chat_button = st.button("New Chat", key="new_chat_button")
                st.markdown('</div>', unsafe_allow_html=True)

                if new_chat_button:
                    # Only save current chat if it has content
                    if st.session_state.chat_history and st.session_state.selected_chat_id is None:
                        # Generate a unique ID for the current chat session based on timestamp
                        chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
                        # Save current chat to file and get updated history
                        updated_history = save_chat_history(chat_id, st.session_state.chat_history)
                        if updated_history:
                            # Update the session state with the latest chat histories
                            st.session_state.all_chat_histories = updated_history
                    
                    # Clear current chat and selection when starting a new chat
                    clear_chat_history()
                    st.session_state.selected_chat_id = None
                    st.rerun()

        # Initialize delete confirmation state if needed
        if "delete_confirm_id" not in st.session_state:
            st.session_state.delete_confirm_id = None

        # Display Chat History in Sidebar, grouped by date
        valid_chats_exist = False
        
        # Ensure all_chat_histories exists
        if "all_chat_histories" not in st.session_state:
            st.session_state.all_chat_histories = load_chat_history_from_json()
        
        # Group chats by date
        grouped_chats = {}
        
        # Debug information
        st.write(f"Debug: Found {len(st.session_state.all_chat_histories)} chats")
        
        for chat_id, chat_data in st.session_state.all_chat_histories.items():
            # Skip invalid chat data
            if not chat_data or not isinstance(chat_data, list) or len(chat_data) == 0:
                continue
                
            # Check the structure of chat entries to accommodate various formats
            first_question = "New Chat"
            found_valid_format = False
            
            # Handle the format: list of tuples like [("You", msg), ("AI", response)]
            if isinstance(chat_data[0], list):
                for item in chat_data[0]:
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "You":
                        first_question = item[1]
                        found_valid_format = True
                        break
            
            # If no valid format was found, try alternative access pattern
            if not found_valid_format and isinstance(chat_data[0], list) and len(chat_data[0]) > 0:
                if isinstance(chat_data[0][0], list) and len(chat_data[0][0]) >= 2:
                    first_question = chat_data[0][0][1]
                    found_valid_format = True
            
            if not found_valid_format:
                # Display debug info about the problematic entry
                st.write(f"Debug: Skipped chat {chat_id} due to format issues")
                st.write(f"Type: {type(chat_data)}, Length: {len(chat_data)}")
                if len(chat_data) > 0:
                    st.write(f"First entry type: {type(chat_data[0])}")
                continue
                
            valid_chats_exist = True
            
            # Parse date from chat_id with improved error handling
            try:
                # Make sure the chat_id is in the expected format
                if chat_id and len(chat_id) >= 14 and chat_id.isdigit():
                    date_obj = datetime.strptime(chat_id, "%Y%m%d%H%M%S")
                    date_str = date_obj.strftime("%d %B %Y")  # Format date as "Day Month Name Year"
                else:
                    # If chat_id doesn't match expected format, try to extract date portion
                    if chat_id and len(chat_id) >= 8 and chat_id[:8].isdigit():
                        # Just extract the date part (first 8 characters: YYYYMMDD)
                        date_part = chat_id[:8]
                        date_obj = datetime.strptime(date_part, "%Y%m%d")
                        date_str = date_obj.strftime("%d %B %Y")
                    else:
                        date_str = "Today"  # Default to "Today" instead of "Unknown Date"
            except ValueError as e:
                # Log the error for debugging
                print(f"Error parsing date from chat_id '{chat_id}': {str(e)}")
                date_str = "Today"  # Use a more user-friendly default

            if date_str not in grouped_chats:
                grouped_chats[date_str] = []
            grouped_chats[date_str].append((chat_id, chat_data, first_question))
        
        # Only display Previous Chats heading if valid chats exist
        if valid_chats_exist:
            st.markdown("<h2 style='text-align: left;font-size:20px;'>PREVIOUS CHATS</h2>", unsafe_allow_html=True)
            
            # Display grouped chats in the sidebar
            for date, chats in grouped_chats.items():
                st.markdown(f"**{date}**")  # Date as a section header
                
                for chat_id, chat_data, first_question in chats:
                    if not first_question or first_question == "":
                        first_question = "New Chat"
                            
                    button_label = f"{first_question[:50]}..." if len(first_question) > 50 else first_question
                    chat_button_key = f"chat_{chat_id}"
                    delete_button_key = f"delete_{chat_id}"
                    confirm_button_key = f"confirm_{chat_id}"
                    cancel_button_key = f"cancel_{chat_id}"
                    
                    # Show delete confirmation UI if this chat is selected for deletion
                    if st.session_state.delete_confirm_id == chat_id:
                        st.markdown(f"""
                        <div class="confirm-dialog">
                            <div style="margin-bottom:8px;">Delete this chat?</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Confirm", key=confirm_button_key):
                                # Perform deletion
                                if delete_chat_history(chat_id):
                                    st.session_state.delete_confirm_id = None
                                    st.toast("Chat deleted!")
                                    st.rerun()
                        with col2:
                            if st.button("Cancel", key=cancel_button_key):
                                st.session_state.delete_confirm_id = None
                                st.rerun()
                    else:
                        # Create a custom clickable row with chat title and delete icon
                        col1, col2 = st.columns([4, 0.5])
                        
                        with col1:
                            # Clickable chat title
                            if st.button(button_label, key=chat_button_key):
                                load_selected_chat(chat_id)
                                st.rerun()
                                
                        with col2:
                            # Delete button
                            if st.button("🗑️", key=delete_button_key):
                                st.session_state.delete_confirm_id = chat_id
                                st.rerun()
        else:
            st.write("No previous chats found.")
            # Show debug information if no valid chats exist
            if len(st.session_state.all_chat_histories) > 0:
                st.write("Debug: Chat history exists but no valid chats were found")
                # Display a sample of the first chat to debug structure
                sample_chat_id = list(st.session_state.all_chat_histories.keys())[0]
                sample_chat = st.session_state.all_chat_histories[sample_chat_id]
                st.write(f"Sample chat ID: {sample_chat_id}")
                st.write(f"Sample chat type: {type(sample_chat)}")
                if isinstance(sample_chat, list) and len(sample_chat) > 0:
                    st.write(f"First entry type: {type(sample_chat[0])}")
                    if len(sample_chat[0]) > 0:
                        st.write(f"First message type: {type(sample_chat[0][0])}")

# Main application function (if needed)
def main():
    # Initialize session state variables if they don't exist
    if "selected_chat_id" not in st.session_state:
        st.session_state.selected_chat_id = None
    
    if "all_chat_histories" not in st.session_state:
        st.session_state.all_chat_histories = load_chat_history_from_json()
    
    # Render the sidebar and chat interface
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()