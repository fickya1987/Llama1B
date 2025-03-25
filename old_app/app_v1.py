# from huggingface_hub import InferenceClient
# import os
# import streamlit as st
# from dotenv import load_dotenv

# load_dotenv()

# client = InferenceClient(api_key=os.getenv('HF_TOKEN'))

# #code below is streamlit code , can ignore if youre not interested in frontend logic 


# st.set_page_config(page_title="💬 SLM-POC")

# with st.sidebar:
#     st.title('💬 Small Language Models - POC')
#     st.write('This chatbot is created using the open-source Llama 3.2 LLM model from Meta, and the Gemma 2b model from Google.')
#     if os.getenv('HF_TOKEN') != None:
#         st.success('API key already provided!', icon='✅')
#         HF_TOKEN = os.getenv('HF_TOKEN')
#         HF_TOKEN = st.secrets['HF_TOKEN']
#     else:
#         HF_TOKEN = st.text_input('Enter HF API token:', type='password')
#         os.environ['HF_TOKEN'] = HF_TOKEN
        
#         if not (HF_TOKEN.startswith('hf_')):
#             st.warning('Please enter your credentials!', icon='⚠️')
#         else:
#             st.success('Proceed to entering your prompt message!', icon='👉')
#     HF_TOKEN = st.secrets['HF_TOKEN']
#     os.environ['HF_TOKEN'] = HF_TOKEN

#     st.subheader('Models and parameters')
#     model_choice = st.selectbox(
#         "Select the Model:" , 
#         ('Llama 3.2 : 1B' , 'Gemma 2 : 2B')
#     )

    
#     temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
#     top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
#     max_length = st.sidebar.slider('max_length', min_value=20, max_value=2040, value=2000, step=5)


# #code above is streamlit code to load HF TOKEN


# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# # disp or clear chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# def clear_chat_history():
#     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
# st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# # user prompt
# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)

# # Generate a new response only if the last message is from the user
# if st.session_state.messages[-1]["role"] == "user":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             if model_choice == 'Gemma 2 : 2B':
#                 clear_chat_history()
#                 stream = client.chat.completions.create(
#                     model="google/gemma-2-2b-it",
#                     messages=st.session_state.messages,
#                     max_tokens=max_length,
#                     temperature=temperature,
#                     top_p=top_p,
#                     stream=True
#                 )
#             elif model_choice == 'Llama 3.2 : 1B':
#                 clear_chat_history()
#                 stream = client.chat.completions.create(
#                     model="meta-llama/Llama-3.2-1B-Instruct",
#                     messages=st.session_state.messages,
#                     max_tokens=max_length,
#                     temperature=temperature,
#                     top_p=top_p,
#                     stream=True
#                 )

#             placeholder = st.empty()
#             full_response = ''
#             for chunk in stream:
#                 if chunk.choices and chunk.choices[0].delta.content:
#                     full_response += chunk.choices[0].delta.content
#                     placeholder.markdown(full_response)

#             # Only append the assistant's response if it was generated successfully
#             if full_response.strip():  # Check if response is not empty
#                 message = {"role": "assistant", "content": full_response}
#                 st.session_state.messages.append(message)
