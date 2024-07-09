import streamlit as st
from chatbot import chat_agent_executor

st.title("Wiki How Demo")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How to Get Popular on Instagram?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chat_agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages})
        result = response["output"]
        st.write(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

if st.button("Clear"):
    st.session_state.messages = []