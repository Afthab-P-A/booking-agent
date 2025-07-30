import streamlit as st
from langchain_core.messages import HumanMessage
from booking_agents import graph  # <- make sure this file has your full agent code

st.set_page_config(page_title="Booking Assistant", layout="centered")

st.title("💇 Booking Assistant Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input prompt
user_prompt = st.chat_input("Type your booking request...")

if user_prompt:
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
    with st.spinner("Thinking..."):
        try:
            response = graph.invoke({"messages": st.session_state.chat_history})
            new_messages = response.get("messages", [])
            st.session_state.chat_history.extend(new_messages)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Show full conversation
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)
