from transformers import pipeline
import streamlit as st

# Model options (key: display name, value: Hugging Face model path)
MODELS = {
    "Default": "distilbert-base-uncased-finetuned-sst-2-english",
    "Twitter-Specific": "cardiffnlp/twitter-roberta-base-sentiment",
    "Multilingual": "nlptown/bert-base-multilingual-uncased-sentiment"
}


@st.cache_resource
def load_model(model_name):
    return pipeline("sentiment-analysis", model=MODELS[model_name])


# Streamlit UI
st.title("Sentiment Analysis App 🔍")
model_choice = st.selectbox("Choose Model", list(MODELS.keys()))
user_input = st.text_area("Your Text", "I love Beer!")

if st.button("Analyze"):
    if user_input:
        model = load_model(model_choice)
        result = model(user_input)[0]

        st.subheader("Result")
        st.write(f"**Model Used:** `{model_choice}`")
        st.write(f"**Label:** {result['label']}")
        st.write(f"**Confidence:** {result['score']:.4f}")
    else:
        st.warning("Please enter text!")