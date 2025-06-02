import datetime
import os
import bs4
from dotenv import load_dotenv

import asyncio
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from typing import AsyncIterable, Any, List

from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult, Document
from langchain_core.prompts import ChatPromptTemplate


# ################################### FastAPI setup ############################################
class Settings(BaseSettings):
    openapi_url: str = ""

settings = Settings()

app = FastAPI(openapi_url=settings.openapi_url)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length"]
)


# ################################### LLM, RAG and Streaming ############################################
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Define the path to your PDF in the repository
PDF_PATH = "nud-handbook.pdf"  # Update this with your actual PDF path

# Standard prompt template for RAG
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant for NU Dasmariñas students. 
Answer the question based ONLY on the following context:
{context}

Question: {question}

Provide a clear, concise, and informative answer:
""")

async def load_pdf_async(file_path: str) -> List[Document]:
    """Load PDF documents asynchronously using PyPDFLoader's alazy_load method"""
    loader = PyPDFLoader(file_path)
    pages = []
    
    # Print some debug info
    print(f"Loading PDF from: {file_path}")
    
    # Use the async lazy_load method as shown in the docs
    async for page in loader.alazy_load():
        pages.append(page)
        
    print(f"Loaded {len(pages)} pages from PDF")
    if pages:
        # Print metadata and a sample of the first page content for debugging
        print(f"First page metadata: {pages[0].metadata}")
        sample = pages[0].page_content[:200] + "..." if len(pages[0].page_content) > 200 else pages[0].page_content
        print(f"Sample content: {sample}")
    
    return pages

async def generate_chunks(query: str, use_pdf: bool = False) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    
    llm = ChatOpenAI(   
        api_key=OPENAI_API_KEY,
        temperature=0.3,  # Lower temperature for more factual responses
        model_name="gpt-3.5-turbo-0125",
        streaming=True,
        verbose=True,
        callbacks=[callback]
    )

    # Choose the loader based on whether to use PDF or web content
    if use_pdf and os.path.exists(PDF_PATH):
        # Use the async PDF loader
        try:
            docs = await load_pdf_async(PDF_PATH)
            print(f"Successfully loaded PDF with {len(docs)} pages")
        except Exception as e:
            print(f"Error loading PDF: {e}")
            # Yield error message and return
            yield f"Sorry, I couldn't load the PDF handbook. Error: {str(e)}"
            return
    else:
        # Use the original web loader
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

    # Use a more aggressive chunking strategy for PDFs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks
        chunk_overlap=50,  # Less overlap
        separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs first
    )
    
    splits = text_splitter.split_documents(docs)
    print(f"Split documents into {len(splits)} chunks")
    
    # For debugging, print a sample chunk
    if splits:
        print(f"Sample chunk: {splits[0].page_content[:100]}...")
    
    try:
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # Retrieve and generate using the relevant snippets of the document
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve more documents for better context
        )
        
        # Use standard prompt
        prompt = RAG_PROMPT

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        task = asyncio.create_task(
            rag_chain.ainvoke(query)
        )
        
        index = 0
        try:
            async for token in callback.aiter():
                print(index, ": ", token, ": ", datetime.datetime.now().time())
                index = index + 1
                yield token
        except Exception as e:
            print(f"Caught exception during streaming: {e}")
            yield f"Sorry, something went wrong while generating the response. Error: {str(e)}"
        finally:
            callback.done.set()

        await task
            
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        yield f"Sorry, an error occurred. Error: {str(e)}"

# ####################### Models ########################################
class Input(BaseModel):
    question: str
    chat_history: list
    use_pdf: bool = False  # Flag to indicate whether to use the PDF

class Metadata(BaseModel):
    conversation_id: str

class Config(BaseModel):
    metadata: Metadata
    
class RequestBody(BaseModel):
    input: Input 
    config: Config

@app.get("/")
def read_root():
    return {
        "status": "API is running",
        "endpoints": ["/chatbot"],
        "version": "1.0.0",
        "pdf_loaded": os.path.exists(PDF_PATH)
    }

@app.post("/chat")
async def chat_redirect(
    query: RequestBody = Body(...),
):
    """Redirect /chat to /chatbot for compatibility"""
    print("Redirecting from /chat to /chatbot")
    return await chat(query)

@app.post("/chatbot")
async def chat(
    query: RequestBody = Body(...),
):
    print(f"Received question: {query.input.question}")
    print(f"Use PDF: {query.input.use_pdf}")
    
    use_pdf = query.input.use_pdf
    
    # Use proper headers for SSE
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
    }
    
    return StreamingResponse(
        generate_chunks(query.input.question, use_pdf),
        media_type="text/event-stream",
        headers=headers
    )
