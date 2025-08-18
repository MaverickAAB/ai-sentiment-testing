import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from transformers import pipeline
from tqdm import tqdm


# Load test data
@st.cache_data
def load_test_data():
    with open("test_data.json") as f:
        return json.load(f)


test_data = load_test_data()

# Models to evaluate
MODELS = {
    "Default": "distilbert-base-uncased-finetuned-sst-2-english",
    "Twitter": "cardiffnlp/twitter-roberta-base-sentiment",
    "3-Star": "finiteautomata/bertweet-base-sentiment-analysis"
}

# Label mappings
LABEL_MAPS = {
    "Default": {"POSITIVE": "positive", "NEGATIVE": "negative"},
    "Twitter": {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"},
    "3-Star": {"POS": "positive", "NEU": "neutral", "NEG": "negative"}
}

# --- Streamlit UI ---
st.title("🧪 **AI Sentiment Analysis Model Comparison**")
"""
**🔍 What this tool does:**  
This dashboard compares 3 sentiment analysis models on 1,500 test cases (including normal text, sarcasm, and emoji-heavy phrases).
Models - 
"Default": "distilbert-base-uncased-finetuned-sst-2-english",
"Twitter": "cardiffnlp/twitter-roberta-base-sentiment",
"3-Star": "finiteautomata/bertweet-base-sentiment-analysis" 
This evaluation uses 1,500 diverse text samples covering:

Standard Sentiment (1,000 samples)
✅ Positive: "This product is amazing!"
😐 Neutral: "It’s okay, not great."
❌ Negative: "Worst experience ever."

Sarcasm (200 samples)
"Oh GREAT, another useless app. 🙄"
"Wow, I LOVE being ignored. 👏"

Emoji-Heavy Text (200 samples)
Positive: "🔥🔥🔥"
Negative: "👎👎👎"

Edge Cases (100 samples)
Minimal text: "Meh."
Ambiguous: "This sentence is false."
Symbols: "@#$%!"

When you click *"Run Evaluation"*, it will:
1. Load all 3 pre-trained models from Hugging Face
2. Process each test case through every model
3. Calculate accuracy metrics and misclassifications
"""

st.markdown("### Evaluating 3 Models on 1500 Test Cases")

# Step 1: Run Evaluations
if st.button("Run Evaluation"):
    results = {"Model": [], "Accuracy": [], "Sarcasm_Accuracy": [], "Emoji_Accuracy": []}
    misclassified_examples = []

    for model_name in MODELS:
        with st.spinner(f"Evaluating {model_name}..."):
            pipe = pipeline("sentiment-analysis", model=MODELS[model_name])
            correct = sarcasm_correct = emoji_correct = 0
            sarcasm_count = emoji_count = 0

            for item in tqdm(test_data):
                text = item["text"]
                expected = item["expected"]
                pred = pipe(text)[0]["label"]
                mapped_pred = LABEL_MAPS[model_name].get(pred, pred).lower()

                # Track general accuracy
                if mapped_pred == expected:
                    correct += 1
                else:
                    misclassified_examples.append({
                        "Model": model_name,
                        "Text": text,
                        "Expected": expected,
                        "Predicted": mapped_pred
                    })

                # Track sarcasm accuracy
                if "🙄" in text or "👌" in text:  # Sarcasm markers
                    sarcasm_count += 1
                    if mapped_pred == expected:
                        sarcasm_correct += 1

                # Track emoji accuracy
                if sum(c in "😀😂😡👍👎" for c in text) >= 3:  # Emoji-heavy
                    emoji_count += 1
                    if mapped_pred == expected:
                        emoji_correct += 1

            results["Model"].append(model_name)
            results["Accuracy"].append(correct / len(test_data))
            results["Sarcasm_Accuracy"].append(sarcasm_correct / max(1, sarcasm_count))
            results["Emoji_Accuracy"].append(emoji_correct / max(1, emoji_count))

    # Save results
    st.session_state.results = pd.DataFrame(results)
    st.session_state.misclassified = pd.DataFrame(misclassified_examples)

# Step 2: Show Results
if "results" in st.session_state:
    st.markdown("---")

    # Metrics Row
    col1, col2, col3 = st.columns(3)
    best_model = st.session_state.results.loc[st.session_state.results["Accuracy"].idxmax(), "Model"]
    col1.metric("Best Model", best_model)
    col2.metric("Highest Accuracy", f"{st.session_state.results['Accuracy'].max():.1%}")
    col3.metric("Test Cases", len(test_data))

    # Accuracy Comparison Chart
    st.markdown("### Model Performance")
    fig, ax = plt.subplots()
    st.session_state.results.set_index("Model").plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

    # Sarcasm/Emoji Breakdown
    st.markdown("### Special Case Performance")
    st.dataframe(st.session_state.results.set_index("Model")[["Sarcasm_Accuracy", "Emoji_Accuracy"]])

    # Misclassified Examples
    st.markdown("### Misclassified Cases")
    model_filter = st.selectbox("Filter by Model", ["All"] + list(MODELS.keys()))
    filtered = st.session_state.misclassified
    if model_filter != "All":
        filtered = filtered[filtered["Model"] == model_filter]
    st.dataframe(filtered.head(20))

    # Download Results
    st.download_button(
        label="Download Full Results (CSV)",
        data=st.session_state.misclassified.to_csv().encode(),
        file_name="misclassified_examples.csv"
    )

else:
    st.info("Click 'Run Evaluation' to start testing models.")