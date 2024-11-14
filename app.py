from huggingface_hub import InferenceClient
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(api_key=os.getenv('HF_TOKEN'))

st.set_page_config(page_title="💬 SLM-POC")

with st.sidebar:
    st.title('💬 Small Language Models - POC')
    st.write('This chatbot is created using various Small Language Models such as Llama 3.2 , Gemma 2 , Phi 3.5 , etc. ')
    if os.getenv('HF_TOKEN') is not None:
        st.success('API key already provided!', icon='✅')
        HF_TOKEN = os.getenv('HF_TOKEN')
        HF_TOKEN = st.secrets['HF_TOKEN']
    else:
        HF_TOKEN = st.text_input('Enter HF API token:', type='password')
        os.environ['HF_TOKEN'] = HF_TOKEN

        if not (HF_TOKEN.startswith('hf_')):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    HF_TOKEN = st.secrets['HF_TOKEN']
    os.environ['HF_TOKEN'] = HF_TOKEN

    st.subheader('Models and parameters')
    model_choice = st.selectbox(
        "Select the Model:",
        ('Llama 3.2 : 1B', 'Phi-3.5' , 'Gemma 2 : 2B')
    )

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=20, max_value=2040, value=2000, step=5)

# init message history for models
if "messages_llama" not in st.session_state:
    st.session_state.messages_llama = []
    st.session_state.messages_llama = [{"role": "assistant", "content": "How may I assist you today?"}]
if "messages_phi" not in st.session_state:
    st.session_state.messages_phi = []
    st.session_state.messages_phi = [{"role": "assistant", "content": "How may I assist you today?"}]
if "messages_gemma" not in st.session_state:
    st.session_state.messages_gemma = []
    st.session_state.messages_gemma = [{"role": "assistant", "content": "How may I assist you today?"}]

# display logic, when model changes -> content changes
if model_choice == 'Llama 3.2 : 1B':
    messages_to_display = st.session_state.messages_llama
elif model_choice =='Phi-3.5':
    messages_to_display = st.session_state.messages_phi
elif model_choice == 'Gemma 2 : 2B':
    messages_to_display = st.session_state.messages_gemma

for message in messages_to_display:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    if model_choice == 'Llama 3.2 : 1B':
        st.session_state.messages_llama = []
    elif model_choice == 'Phi-3.5':
        st.session_state.messages_phi = []
    elif model_choice == 'Gemma 2 : 2B':
        st.session_state.messages_gemma = []
        
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# user prompt logic
if prompt := st.chat_input():
    if model_choice == 'Llama 3.2 : 1B':
        st.session_state.messages_llama.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    elif model_choice == 'Phi-3.5':
        st.session_state.messages_phi.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    elif model_choice =='Gemma 2 : 2B':
        st.session_state.messages_gemma.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)


# gen a new response from slm if last msg is from user
if model_choice == 'Llama 3.2 : 1B' and st.session_state.messages_llama and st.session_state.messages_llama[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = client.chat.completions.create(
                model="meta-llama/Llama-3.2-1B-Instruct",
                messages=st.session_state.messages_llama,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )

            placeholder = st.empty()
            full_response = ''
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response)

            if full_response.strip():
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages_llama.append(message)

elif model_choice == 'Phi-3.5' and st.session_state.messages_phi and st.session_state.messages_phi[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = client.chat.completions.create(
                model="microsoft/Phi-3.5-mini-instruct",
                messages=st.session_state.messages_phi,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )

            placeholder = st.empty()
            full_response = ''
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response)

            if full_response.strip():
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages_phi.append(message)

elif model_choice == 'Gemma 2 : 2B' and st.session_state.messages_gemma and st.session_state.messages_gemma[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = client.chat.completions.create(
                model="google/gemma-1.1-2b-it",
                messages=st.session_state.messages_gemma,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )

            placeholder = st.empty()
            full_response = ''
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response)

            if full_response.strip():
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages_gemma.append(message)