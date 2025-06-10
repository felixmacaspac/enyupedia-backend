# NU Dasmariñas Chatbot

A conversational AI assistant built specifically for NU Dasmariñas students. This chatbot uses Retrieval Augmented Generation (RAG) to provide accurate information from the NU Dasmariñas handbook.

## Features

- **Intelligent Q&A**: Ask questions about NU Dasmariñas policies, procedures, and guidelines
- **PDF Knowledge Base**: Powered by the official NU Dasmariñas handbook
- **Real-time Responses**: Streaming responses for a natural conversation experience
- **Mobile and Web Access**: Available through React Native mobile app and web interface

## Tech Stack

### Backend

- **FastAPI**: Modern, high-performance API framework
- **LangChain**: Framework for LLM application development
- **OpenAI API**: Powers the underlying language model (GPT-3.5 Turbo)
- **Chroma Vector DB**: Stores document embeddings for semantic search
- **Uvicorn**: ASGI server for FastAPI

### Frontend

- **React Native**: For the mobile application interface
- **Streamlit**: For the web application interface
