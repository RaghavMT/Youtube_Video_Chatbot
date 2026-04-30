from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from typing import Optional, Dict, Any
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage, SystemMessage

#Step 1A Indexing (document ingestion)
# 1. Assume the URL is fetched/passed from elsewhere in app.py

def get_youtube_transcript_runnable(input_data: Dict[str, Any]) -> Dict[str, Any]:
    url = input_data.get("url", "")
    pattern = r'(?:v=|\/|be\/|embed\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    video_id = match.group(1) if match else "LNHBMFCzznE"

    try:
        # Step 1: Attempt to get the transcript
        # We try the standard way first
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        except:
            # Fallback for instance-based versions
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id).find_transcript(['en']).fetch()

        # Step 2: Extract text safely (Fixes 'Not Subscriptable')
        final_text_parts = []
        for chunk in transcript_list:
            # Check if it's a dictionary: chunk['text']
            if isinstance(chunk, dict):
                final_text_parts.append(chunk.get('text', ''))
            # Check if it's an object: chunk.text
            elif hasattr(chunk, 'text'):
                final_text_parts.append(getattr(chunk, 'text'))
            # Last resort: convert the whole chunk to string
            else:
                final_text_parts.append(str(chunk))

        transcript_text = " ".join(final_text_parts)

        return {
            "transcript": transcript_text,
            "video_id": video_id,
            "error": None
        }

    except Exception as e:
        return {"transcript": None, "video_id": video_id, "error": str(e)}

# Convert to a LangChain Runnable
transcript_chain = RunnableLambda(get_youtube_transcript_runnable)


# Example Usage within a larger chain:
# chain = transcript_chain | prompt | model | output_parser
# result = chain.invoke({"url": "https://www.youtube.com/watch?v=your_id"})

#Step 1B - text Splitting
def splitting(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the output from transcript_chain and splits the text into chunks.
    """
    if input_data.get("error"):
            return {"chunks": [], "error": input_data.get("error")}
    text_to_split = input_data.get("transcript", "")

    if not text_to_split:
        return {"chunks":[], "error": input_data.get("error", "No transcript found")}
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text_to_split)

    return{"chunks": chunks}

chunk_chain = RunnableLambda(splitting)

#Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
def vector_store_func(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts strings to documents and creates the FAISS vectore store
    """

    if input_data.get("error"):
        return {"vector_store": None, "error": input_data.get("error")}
    
    chunks = input_data.get("chunks", [])

    if not chunks:
        return{"vector_store" : None, "error": "No chunks found"}

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
    try:
        vector_db = FAISS.from_texts(chunks, embeddings)
        return {"vector_store": vector_db}
    except Exception as e:
        return {"vector_store": None, "error": str(e)}

vector_chain = RunnableLambda(vector_store_func)

#Step 2 Retrieval
def create_retriever_runnable(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts the FAISS vector store into a retriever.
    Input: {"vector_store": FAISS_Object}
    Output: {"retriever": Retriever_Object}
    """

    if input_data.get("error"):
        return {"retriever": None, "error": input_data.get("error")}

    vs = input_data.get("vector_store")
    
    if not vs:
        return {"retriever": None, "error": "No vector store found to create retriever"}

    retriever = vs.as_retriever(search_type="similarity",search_kwargs={"k": 4})

    return {"retriever": retriever}

retriever_chain = RunnableLambda(create_retriever_runnable)


ingestion_chain = transcript_chain | chunk_chain | vector_chain | retriever_chain

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      Context : {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

p_llm = ChatPollinations(
    model = "nova-fast",
    temperature = 0,
    api_key="sk_D2GhsbL16wAKOcsOL6ki4fdEFsZmRENr"
)

