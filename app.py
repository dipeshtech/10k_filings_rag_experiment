import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv

load_dotenv()


# provide the path of  pdf file/files.
pdfreader = PdfReader('input_data/nvidia_10k.pdf')

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

@st.cache_data
def split_chunk_text(input_path="input_data/nvidia_10k.pdf"):
    from typing_extensions import Concatenate
    # read text from pdf
    
    pdfreader = PdfReader(input_path)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    
    return texts
    





with st.form("my_form"):
    texts = split_chunk_text()
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
        )
    
    vector_store = FAISS.from_texts(texts, embeddings)
    retriever = vector_store.as_retriever()
    text = st.text_area("Enter question:", " ")
    
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        #texts = split_chunk_text()
        
        from operator import itemgetter

        from langchain.prompts import ChatPromptTemplate

        template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

        Context:
        {context}

        Question:
        {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        retrieval_augmented_qa_chain = (
            # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
            # "question" : populated by getting the value of the "question" key
            # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
            #              by getting the value of the "context" key from the previous step
            | RunnablePassthrough.assign(context=itemgetter("context"))
            # "response" : the "context" and "question" values are used to format our prompt object and then piped
            #              into the LLM and stored in a key called "response"
            # "context"  : populated by getting the value of the "context" key from the previous step
            | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
        )
        #query = "Who is liable in case of an accident if a learner is driving with an instructor?"
        
        result = retrieval_augmented_qa_chain.invoke({"question" : text})
        
        
        st.info(result["response"].content)