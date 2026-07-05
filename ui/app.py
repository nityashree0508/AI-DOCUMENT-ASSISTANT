import streamlit as st
import requests

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

st.sidebar.title("🤖 AI Assistant")

st.sidebar.success("Backend Connected")

st.sidebar.markdown("---")

st.sidebar.write("Powered by")

st.sidebar.write("• LangGraph")

st.sidebar.write("• Gemini 2.5 Flash")

st.sidebar.write("• FAISS")

st.sidebar.write("• Hybrid Retrieval")

st.sidebar.write("• Cross Encoder")

st.title("📄 AI Document Assistant")

st.write("Ask questions about your uploaded research paper.")

question = st.text_input(
    "Question"
)

if st.button("Ask"):

    with st.spinner("Searching document..."):

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={
                "question": question
            }
        )

    answer = response.json()["answer"]

    st.markdown("## Answer")

    st.success(answer)