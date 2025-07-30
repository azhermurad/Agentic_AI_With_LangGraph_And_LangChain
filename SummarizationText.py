import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from pydantic import BaseModel, HttpUrl, ValidationError
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat


# Load API key from .env
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# 🧠 Custom callback handler for ChatGPT-style word-by-word streaming
class ChatGPTStyleStreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.output = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.output += token
        self.placeholder.markdown(self.output)  # Add cursor feel


# Streamlit UI
st.set_page_config(page_title="ChatGPT-Style Summarizer")
st.title("💬 LangChain: Summarize Text From Youtube or Website")


# URL input
class URLModel(BaseModel):
    website: HttpUrl


url = st.text_input(
    "Enter a URL to  summarize", "https://python.langchain.com/docs/expression_language"
)


if st.button("Summarize"):

    if not url:
        st.error("Please enter a URL.")
        st.stop()
    else:
        try:
            # Try with an invalid URL
            url = URLModel(website=url).website
            url = str(url)
        except ValidationError as e:
            st.error("Invalid URL. Please enter a valid URL.")
            st.stop()

    with st.spinner("🔄 Loading content..."):
        if "youtube.com" in url:
            # pass
            loader = YoutubeLoader.from_youtube_url(
                "https://www.youtube.com/watch?v=TKCMw0utiak",
                add_video_info=True,
                # transcript_format=TranscriptFormat.CHUNKS,
                # chunk_size_seconds=30,
            )

            print("youtube")
            docs = loader.load()
            st.write(docs)
            st.stop()

        else:
            print("website")
            loader = WebBaseLoader(url)

    #     st.write("continue")
    docs = loader.load()
    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Summarize clearly and concisely.",
            ),
            (
                "human",
                "Summarize the following:\n\n{context}",
            ),
        ]
    )

    # Placeholder to update live output like ChatGPT
    output_placeholder = st.empty()
    callback_handler = ChatGPTStyleStreamHandler(output_placeholder)

    # Chat model with streaming
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        streaming=True,
        temperature=0.3,
        # callbacks=[callback_handler],
    )

    # Create summarization chain

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)

    # st.write(all_splits)
    # chain.invoke({"context": all_splits})

    # chain = create_stuff_documents_chain(llm, prompt)
    # chain.invoke({"context": docs})

    # Map-Reduce: summarize long texts via parallelization

    map_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
You are a professional research assistant.

Below is an information from a webpage . Read it carefully and summarize its **main ideas and key points** in 3 to 5 bullet points.

Use clear, concise language and focus only on the information in the information.

`========
{text}
========`

Summary in bullet points:
""",
    )
    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
You are an expert summarizer.

Below are several summaries of different sections of a webpage. Your task is to combine them into a single, cohesive summary in 5 to 7 bullet points.

Avoid repetition, eliminate redundant information, and ensure clarity and completeness.

---

{text}

---

Final Summary:
""",
    )

    # chain = load_summarize_chain(
    #     llm=llm,
    #     chain_type="map_reduce",
    #     map_prompt=map_prompt,
    #     combine_prompt=combine_prompt,
    #     verbose=True,
    # )

    # refine chain summarize
    refinechain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        verbose=True,
    )

    st.write("\n✅ Final Summary:\n", refinechain.invoke(all_splits)["output_text"])
