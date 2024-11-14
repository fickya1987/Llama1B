from huggingface_hub import InferenceClient
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()


client = InferenceClient(api_key=os.getenv('HF_TOKEN'))

st.set_page_config(page_title="🦙💬 Llama 3.2 Chatbot")

with st.sidebar:
    with st.expander('🦙💬 Llama 3.2 Chatbot'):
        st.write('Model Information')
    st.write('This chatbot is created using the open-source Llama 3.2 : 1B LLM model from Meta.')
    if os.getenv('HF_TOKEN') != None:
        st.success('API key already provided!', icon='✅')
    else:
        HF_TOKEN = st.text_input('Enter HF API token:', type='password')
        os.environ['HF_TOKEN'] = HF_TOKEN
        
        if not (HF_TOKEN.startswith('hf_')):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=20, max_value=2040, value=100, step=5)


    # uploading doc functionality
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:

        # need temp file to store doc, as LangChain doesnt support streamlit doc upload

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # load based on type of doc
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # IMP # ideally use OpenAI embeddings, I have used others, due to API key limitation, but due to LangChain error, they fall back on a dummy tokenizer
        embeddings = HuggingFaceHubEmbeddings(huggingfacehub_api_token = os.getenv('HF_TOKEN'))
        vector_store = FAISS.from_documents(texts, embeddings)

        

        st.success("Document uploaded and processed successfully!")

    

        #directly taken chain from lang chain
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            max_length = max_length,
            temperature=temperature,
            top_p=top_p,
            huggingfacehub_api_token=os.getenv('HF_TOKEN'),
            )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
            )

#  session state for messages
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# o/p chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)



# user prompt goes here
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", 
                                      "content": " You are a Chatbot You have to answer questions based on your knowledge or if the prompt is relevant to the document uploaded, answer based on document. Provide a confidence score with every response. Prompt is: {prompt}"})
    with st.chat_message("user"):
        st.write(prompt)

    # logic to check if the last message is from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if uploaded_file is not None:

                    #  RAG chain is called here
                    response = qa_chain({"query": prompt})
                    full_response = response['result']
                else:
                    # if no doc uploaded, chat normally 

                    stream = client.chat.completions.create(
                        model="meta-llama/Llama-3.2-1B-Instruct",
                        messages=st.session_state.messages,
                        max_tokens=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        stream=True
                    )
                    
                    placeholder = st.empty()
                    full_response = ''

                    # o/p streaming
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            placeholder.markdown(full_response)

                # add response to state
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)