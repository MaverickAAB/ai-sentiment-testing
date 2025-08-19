import streamlit as st
import pandas as pd
import plotly.express as px
import json
from transformers import pipeline
from tqdm import tqdm

# Initialize session state
if 'evaluation_done' not in st.session_state:
    st.session_state.evaluation_done = False
if 'results' not in st.session_state:
    st.session_state.update({
        'results': None,
        'misclassified': None,
        'evaluation_done': False,
        'test_data': None
    })

# Load test data
@st.cache_data
def load_test_data():
    try:
        with open('test_data.json') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load test data: {str(e)}")
        return None

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

# Evaluation function
def run_evaluation():
    if st.session_state.test_data is None:
        st.error("No test data loaded!")
        return

    results = {"Model": [], "Accuracy": [], "Sarcasm_Accuracy": [], "Emoji_Accuracy": []}
    misclassified = []

    for model_name in MODELS:
        with st.spinner(f"Evaluating {model_name}..."):
            pipe = pipeline("sentiment-analysis", model=MODELS[model_name])
            correct = sarcasm_correct = emoji_correct = 0
            sarcasm_count = emoji_count = 0

            for item in tqdm(st.session_state.test_data):
                text = item["text"]
                expected = item["expected"]
                result = pipe(text)[0]
                pred = result["label"]
                mapped_pred = LABEL_MAPS[model_name].get(pred, pred).lower()

                if mapped_pred == expected:
                    correct += 1
                    if "🙄" in text or "👌" in text:
                        sarcasm_correct += 1
                    if sum(c in "😀😂😡👍👎" for c in text) >= 3:
                        emoji_correct += 1
                else:
                    misclassified.append({
                        "Model": model_name,
                        "Text": text,
                        "Expected": expected,
                        "Predicted": mapped_pred,
                        "Confidence": result["score"]
                    })

                if "🙄" in text or "👌" in text:
                    sarcasm_count += 1
                if sum(c in "😀😂😡👍👎" for c in text) >= 3:
                    emoji_count += 1

            results["Model"].append(model_name)
            results["Accuracy"].append(correct / len(st.session_state.test_data))
            results["Sarcasm_Accuracy"].append(sarcasm_correct / max(1, sarcasm_count))
            results["Emoji_Accuracy"].append(emoji_correct / max(1, emoji_count))

    st.session_state.results = pd.DataFrame(results)
    st.session_state.misclassified = pd.DataFrame(misclassified)
    st.session_state.evaluation_done = True

# Show results function
def show_results():
    if st.session_state.results is None:
        st.warning("No results available. Run evaluation first.")
        return

    st.subheader("📊 Performance Overview")
    cols = st.columns(3)
    best_model = st.session_state.results.loc[st.session_state.results["Accuracy"].idxmax(), "Model"]
    cols[0].metric("Best Model", best_model)
    cols[1].metric("Highest Accuracy", f"{st.session_state.results['Accuracy'].max():.1%}")
    cols[2].metric("Test Cases", len(st.session_state.test_data))

    st.subheader("🔍 Performance by Text Type")
    plot_df = st.session_state.results.rename(columns={
        "Accuracy": "Overall",
        "Sarcasm_Accuracy": "Sarcasm",
        "Emoji_Accuracy": "Emoji"
    }).copy()
    plot_df = plot_df.melt(
        id_vars=["Model"],
        value_vars=["Overall", "Sarcasm", "Emoji"],
        var_name="Category",
        value_name="Accuracy Score"
    )
    fig = px.bar(
        plot_df,
        x="Model",
        y="Accuracy Score",
        color="Category",
        barmode="group",
        text_auto=".1%",
        height=500,
        labels={"Accuracy Score": "Accuracy"}
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit UI (Main App Flow) ---
st.set_page_config(layout="wide")
st.title("🧠 Sentiment Analysis Model Comparison")
st.markdown("Performance comparison using 1500 test cases across 3 sentiment analysis models")

if st.session_state.test_data is None:
    st.session_state.test_data = load_test_data()

if st.button("▶️ Run Evaluation", type="primary"):
    run_evaluation()

if st.session_state.get('evaluation_done', False):
    show_results()

    st.subheader("❌ Misclassified Cases")
    model_filter = st.selectbox("Filter by model:", ["All"] + list(MODELS.keys()))

    filtered = st.session_state.misclassified
    if model_filter != "All":
        filtered = filtered[filtered["Model"] == model_filter]
    st.dataframe(
        filtered.sort_values("Confidence", ascending=False).head(20),
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Full Results",
        data=st.session_state.misclassified.to_csv(index=False).encode(),
        file_name="sentiment_results.csv",
        mime="text/csv"
    )