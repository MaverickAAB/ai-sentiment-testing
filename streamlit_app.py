import streamlit as st
import pandas as pd
import plotly.express as px
import json
from transformers import pipeline
from tqdm import tqdm
import torch
from typing import List, Dict, Optional
import time

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

# Constants
BATCH_SIZE = 32  # Optimal for most GPU/CPU scenarios
MODELS = {
    "Default": "distilbert-base-uncased-finetuned-sst-2-english",
    "Twitter": "cardiffnlp/twitter-roberta-base-sentiment",
    "3-Star": "finiteautomata/bertweet-base-sentiment-analysis"
}
LABEL_MAPS = {
    "Default": {"POSITIVE": "positive", "NEGATIVE": "negative"},
    "Twitter": {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"},
    "3-Star": {"POS": "positive", "NEU": "neutral", "NEG": "negative"}
}

# Device configuration (automatically detects GPU)
DEVICE = 0 if torch.cuda.is_available() else -1


@st.cache_data
def load_test_data() -> Optional[List[Dict]]:
    """Load and validate test data from JSON file."""
    try:
        with open('test_data.json') as f:
            data = json.load(f)
            if not isinstance(data, list):
                st.error("Test data should be a list of test cases")
                return None
            return data
    except Exception as e:
        st.error(f"Failed to load test data: {str(e)}")
        return None


@st.cache_resource  # Cache models to avoid reloading
def load_model(model_name: str):
    """Load and cache the sentiment analysis model."""
    return pipeline(
        "sentiment-analysis",
        model=MODELS[model_name],
        device=DEVICE,
        truncation=True,
        padding=True
    )


def run_evaluation():
    if st.session_state.test_data is None:
        st.error("No test data loaded!")
        return

    test_data = st.session_state.test_data
    results = {"Model": [], "Accuracy": [], "Sarcasm_Accuracy": [], "Emoji_Accuracy": []}
    misclassified = []

    # Display device being used
    device_name = "GPU 🔥" if DEVICE != -1 else "CPU 🐢"
    st.info(f"Running on: {device_name} | Batch size: {BATCH_SIZE}")

    progress_bar = st.progress(0)
    status_text = st.empty()
    time_log = []

    for model_idx, model_name in enumerate(MODELS):
        start_time = time.time()
        status_text.text(f"Evaluating {model_name}...")

        try:
            pipe = load_model(model_name)
            texts = [item["text"] for item in test_data]
            expected = [item["expected"] for item in test_data]

            batch_stats = {
                'correct': 0,
                'sarcasm_correct': 0,
                'emoji_correct': 0,
                'sarcasm_count': 0,
                'emoji_count': 0,
                'misclassified': []
            }

            # Process in batches
            for i in tqdm(range(0, len(texts), BATCH_SIZE)):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_expected = expected[i:i + BATCH_SIZE]

                try:
                    predictions = pipe(batch_texts, batch_size=BATCH_SIZE)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        st.warning("Reducing batch size due to memory constraints")
                        predictions = pipe(batch_texts, batch_size=BATCH_SIZE // 2)
                    else:
                        raise e

                for pred, text, exp in zip(predictions, batch_texts, batch_expected):
                    pred_label = LABEL_MAPS[model_name].get(pred["label"], pred["label"]).lower()
                    has_sarcasm = "🙄" in text or "👌" in text
                    has_emoji = sum(c in "😀😂😡👍👎" for c in text) >= 3

                    if pred_label == exp:
                        batch_stats['correct'] += 1
                        if has_sarcasm: batch_stats['sarcasm_correct'] += 1
                        if has_emoji: batch_stats['emoji_correct'] += 1
                    else:
                        batch_stats['misclassified'].append({
                            "Model": model_name,
                            "Text": text,
                            "Expected": exp,
                            "Predicted": pred_label,
                            "Confidence": pred["score"]
                        })

                    if has_sarcasm: batch_stats['sarcasm_count'] += 1
                    if has_emoji: batch_stats['emoji_count'] += 1

                # Update progress
                progress = (i / len(texts)) * (1 / len(MODELS)) + (model_idx / len(MODELS))
                progress_bar.progress(min(progress, 1.0))

            # Store results
            results["Model"].append(model_name)
            results["Accuracy"].append(batch_stats['correct'] / len(test_data))
            results["Sarcasm_Accuracy"].append(
                batch_stats['sarcasm_correct'] / max(1, batch_stats['sarcasm_count']))
            results["Emoji_Accuracy"].append(
                batch_stats['emoji_correct'] / max(1, batch_stats['emoji_count']))
            misclassified.extend(batch_stats['misclassified'])

            # Log timing
            elapsed = time.time() - start_time
            time_log.append(f"{model_name}: {elapsed:.2f}s")

        except Exception as e:
            st.error(f"Error evaluating {model_name}: {str(e)}")
            continue

    st.session_state.results = pd.DataFrame(results)
    st.session_state.misclassified = pd.DataFrame(misclassified)
    st.session_state.evaluation_done = True

    progress_bar.empty()
    status_text.text("Evaluation complete!")
    st.balloons()

    # Show performance metrics
    with st.expander("⚡ Performance Metrics"):
        st.write("Evaluation times:")
        st.write("\n".join(time_log))
        if DEVICE != -1:
            st.write(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB used")


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


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🧠 Sentiment Analysis Model Comparison")
st.markdown("Performance comparison using test cases across 3 sentiment analysis models with 1500 test cases")

with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    - **Default**: General purpose sentiment analysis  
    - **Twitter**: Optimized for Twitter/social media content  
    - **3-Star**: Handles 3-class sentiment (positive/neutral/negative)
    """)
    if DEVICE != -1:
        st.success(f"Using GPU acceleration ({torch.cuda.get_device_name(0)})")
    else:
        st.warning("Running on CPU - consider using GPU for faster performance")

if st.session_state.test_data is None:
    st.session_state.test_data = load_test_data()

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Run Evaluation", type="primary"):
        run_evaluation()
with col2:
    if st.session_state.get('evaluation_done', False):
        if st.button("🔄 Reset Evaluation"):
            st.session_state.evaluation_done = False
            st.session_state.results = None
            st.session_state.misclassified = None
            st.rerun()

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