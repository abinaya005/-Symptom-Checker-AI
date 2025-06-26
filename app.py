import streamlit as st
from transformers import pipeline

# Load a question-answering pipeline with a medical model
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="deepset/bert-base-cased-squad2")

qa_model = load_model()

# Sample medical context (in a real app, this can be replaced by MedQuAD, PubMed, etc.)
context = """
Fever is the body's natural response to infection. It can be caused by bacterial or viral infections.
Common symptoms include sore throat, fatigue, headache, and muscle aches.
Headaches can occur due to stress, dehydration, eye strain, or viral illness.
Dizziness may result from inner ear problems, low blood pressure, or dehydration.
"""


st.title("🩺 Free Symptom Checker AI (No API Needed)")

question = st.text_input("Enter your symptom or health question:")

if question:
    with st.spinner("Analyzing..."):
        result = qa_model(question=question, context=context)
        st.markdown("### AI Answer:")
        if result["score"] > 0.3:
            st.write(result["answer"])
        else:
            st.write("Sorry, I couldn't find a reliable answer. Please consult a doctor.")
