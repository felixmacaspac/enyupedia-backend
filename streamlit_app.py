import streamlit as st
import requests
import uuid

st.title("EnyuPedia - NU Dasmariñas Knowledge Base Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Generate a conversation ID if not present
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if prompt := st.chat_input("Ask something about NU Dasmariñas..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    api_url = "http://localhost:8000/chatbot"  
    payload = {
        "input": {
            "question": prompt,
            "chat_history": st.session_state.messages,
            "use_pdf": True 
        },
        "config": {
            "metadata": {
                "conversation_id": st.session_state.conversation_id
            }
        }
    }
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with requests.post(api_url, json=payload, stream=True) as r:
                r.raise_for_status() 
                
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk_text = line.decode('utf-8', errors='replace')
                            full_response += chunk_text
                            message_placeholder.markdown(full_response + "▌")
                        except Exception as decode_error:
                            st.error(f"Error decoding response: {decode_error}")
                            continue
            
            message_placeholder.markdown(full_response)
        except requests.exceptions.RequestException as e:
            error_message = f"Error connecting to the API: {str(e)}"
            message_placeholder.error(error_message)
            st.error(f"Details: {e}")
            full_response = error_message
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
