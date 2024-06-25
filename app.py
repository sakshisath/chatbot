import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import HarmCategory, HarmBlockThreshold

def multiturn_generate_content(user_input, generation_config, safety_settings, chat_history):
    vertexai.init(project="flash-aviary-426023-j0", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-001")
    chat = model.start_chat()
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        return "Goodbye!", chat_history
    
    # Append the user input to the chat history
    chat_history.append({"role": "user", "content": user_input})
    
    # Prepare messages for the model
    messages = [entry["content"] for entry in chat_history]

    bot_response = chat.send_message(
        messages,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Extract the text response from the bot's response
    bot_response_text = bot_response.candidates[0].content.parts[0].text
    
    # Append the bot response to the chat history
    chat_history.append({"role": "bot", "content": bot_response_text})
    
    return bot_response_text, chat_history

def main():
    st.title("Welcome to Chat!")
    st.write("How Can I Help You?")

    # Initialize chat history 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    chat_placeholder = st.empty()
    chat_history_text = "\n\n".join(
        [f"**You**: {entry['content']}" if entry['role'] == 'user' else f"**Chatbot**: {entry['content']}" for entry in st.session_state.chat_history]
    )
    chat_placeholder.markdown(chat_history_text)

    # User input at the bottom
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send"):
        # Define generation config and safety settings
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Call chatbot function with parameters and current chat history
        bot_response, st.session_state.chat_history = multiturn_generate_content(
            user_input, generation_config, safety_settings, st.session_state.chat_history
        )

        # Clear the text input by rerunning the script
        st.experimental_rerun()

if __name__ == "__main__":
    main()
