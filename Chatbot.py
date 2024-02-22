
import streamlit as st
import docx2txt
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.schema import SystemMessage
from PyPDF2 import PdfReader
from langchain_community.vectorstores import faiss
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain 
from langchain.memory import ConversationBufferMemory
load_dotenv()

#Here the user inputs something, and the AI returns with a message after being fed the files
def get_conversation_chain(context):
    llm = ChatOpenAI(temperature=0, )
    messages = ChatPromptTemplate.from_messages([
        SystemMessage(content=context),
        MessagesPlaceholder(
            variable_name = "chat_history"
        ),
        HumanMessagePromptTemplate.from_template(
            "{question}"
        ),
    ])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = LLMChain(
        llm=llm,
        prompt=messages,
        memory=memory,
        verbose = True
    )
    return chain

#Gets all relevant data from the database, produces the context for the AI
def get_data_vectorstore(question,vector_db):
    docs = vector_db.similarity_search(str(question))
    context = "You are helpful assistant, answer all questions to the best of your ability from this information:" 
    context += str(docs)
    return context

#PDF is extracted into text document so it is readable for the AI 
def save_data_vectorstore(files):
    text = ''
    for file in files:
        ext = file.name.rsplit('.', 1)[-1]
        if ext == 'pdf':
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        if ext == 'docx':
            text += docx2txt.process(file)
    doc =  Document(page_content=text, metadata={"source": "local"})
    embeddings = OpenAIEmbeddings()
    vector_db = faiss.FAISS.from_documents([doc],embeddings)
    return vector_db

#the main program
def main():
    st.header('Demo AI')
    files = st.file_uploader(label="Choose file:",type=['pdf','docx'], accept_multiple_files=True)
    save = st.button('Save')
    question = st.text_input('Input User')
    # creates session states
    if 'conversations' not in st.session_state:
        st.session_state.conversations = []
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = save_data_vectorstore(files)
    if save:
        st.session_state.vector_db = save_data_vectorstore(files)
    # submit button that gets the uploaded files
    if st.button('Submit'):
        docs = get_data_vectorstore(question,st.session_state.vector_db)
        if 'chain' not in st.session_state:
            st.session_state.chain = get_conversation_chain(context=docs)
        answer = st.session_state.chain({"question":question})
        st.session_state.conversations.append({'user':question})
        st.session_state.conversations.append({'assistant':answer['text']})
    # to dogit
    for conversation in st.session_state.conversations:
        for key,value in conversation.items():
            if key == 'user':
                user = st.chat_message("user")
                user.write(value)
            if key == 'assistant':
                assistant = st.chat_message("assistant")
                assistant.write(value) 
        
if __name__ == "__main__":
    main()


