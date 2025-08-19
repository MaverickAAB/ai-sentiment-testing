import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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
st.set_page_config(layout="wide")
st.title("🧠 Sentiment Analysis Model Comparison Dashboard")
st.markdown("""
**🔍 What this dashboard shows:**
- Performance comparison using 1500 test cases across 3 sentiment analysis models
  - Default: distilbert-base-uncased-finetuned-sst-2-english
  - Twitter: cardiffnlp/twitter-roberta-base-sentiment
  - 3-Star: finiteautomata/bertweet-base-sentiment-analysis
- Specialized test analysis for different text types (sarcasm, emoji-heavy, etc.)
- Detailed breakdown of model strengths and weaknesses
- Takes about 5 mins on my local machine. For more details, check README
""")


# --- Evaluation Function ---
def run_evaluation():
    results = {"Model": [], "Accuracy": [], "Sarcasm_Accuracy": [], "Emoji_Accuracy": [], "Edge_Case_Accuracy": []}
    misclassified_examples = []
    case_analysis = []

    for model_name in MODELS:
        with st.spinner(f"Evaluating {model_name}..."):
            pipe = pipeline("sentiment-analysis", model=MODELS[model_name])
            correct = sarcasm_correct = emoji_correct = edge_correct = 0
            sarcasm_count = emoji_count = edge_count = 0

            for item in tqdm(test_data):
                text = item["text"]
                expected = item["expected"]
                result = pipe(text)[0]
                pred = result["label"]
                confidence = result["score"]
                mapped_pred = LABEL_MAPS[model_name].get(pred, pred).lower()

                # Determine case type
                case_type = "Standard"
                if "🙄" in text or "👌" in text:
                    case_type = "Sarcasm"
                    sarcasm_count += 1
                elif sum(c in "😀😂😡👍👎" for c in text) >= 3:
                    case_type = "Emoji-Heavy"
                    emoji_count += 1
                elif len(text.split()) <= 2:
                    case_type = "Edge Case"
                    edge_count += 1

                # Track accuracy
                is_correct = mapped_pred == expected
                if is_correct:
                    correct += 1
                    if case_type == "Sarcasm":
                        sarcasm_correct += 1
                    elif case_type == "Emoji-Heavy":
                        emoji_correct += 1
                    elif case_type == "Edge Case":
                        edge_correct += 1
                else:
                    misclassified_examples.append({
                        "Model": model_name,
                        "Text": text,
                        "Expected": expected,
                        "Predicted": mapped_pred,
                        "Confidence": confidence,
                        "Case Type": case_type
                    })

                case_analysis.append({
                    "Model": model_name,
                    "Text": text,
                    "Case Type": case_type,
                    "Correct": is_correct
                })

            # Store results
            results["Model"].append(model_name)
            results["Accuracy"].append(correct / len(test_data))
            results["Sarcasm_Accuracy"].append(sarcasm_correct / max(1, sarcasm_count))
            results["Emoji_Accuracy"].append(emoji_correct / max(1, emoji_count))
            results["Edge_Case_Accuracy"].append(edge_correct / max(1, edge_count))

    # Save results
    st.session_state.results = pd.DataFrame(results)
    st.session_state.misclassified = pd.DataFrame(misclassified_examples)
    st.session_state.case_analysis = pd.DataFrame(case_analysis)


# --- Visualization Functions ---
def plot_category_performance():
    st.subheader("📊 Model Performance by Text Category")

    # Create a copy of results to avoid modifying the original
    plot_df = st.session_state.results.copy()

    # Rename columns to remove "_Accuracy" suffix for cleaner display
    plot_df = plot_df.rename(columns={
        "Sarcasm_Accuracy": "Sarcasm",
        "Emoji_Accuracy": "Emoji-Heavy",
        "Edge_Case_Accuracy": "Edge Cases",
        "Accuracy": "Overall"
    })

    # Melt dataframe for Plotly
    plot_df = plot_df.melt(
        id_vars=["Model"],
        value_vars=["Overall", "Sarcasm", "Emoji-Heavy", "Edge Cases"],
        var_name="Category",
        value_name="Accuracy Score"  # Changed from "Accuracy" to avoid conflict
    )

    fig = px.bar(
        plot_df,
        x="Model",
        y="Accuracy Score",
        color="Category",
        barmode="group",
        text_auto=".1%",
        height=500,
        title="Accuracy Across Different Text Categories",
        labels={"Accuracy Score": "Accuracy", "Model": ""},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend_title_text="Text Category"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_case_examples():
    st.subheader("🔍 Case Analysis")

    # Summary stats
    case_summary = st.session_state.case_analysis.groupby(
        ["Model", "Case Type"]
    )["Correct"].mean().unstack().style.format("{:.1%}").background_gradient(cmap="Blues")

    st.markdown("### Accuracy by Case Type")
    st.dataframe(case_summary, use_container_width=True)

    # Detailed examples
    with st.expander("View Detailed Examples", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Sarcasm", "Emoji-Heavy", "Edge Cases"])

        for case_type, tab in [("Sarcasm", tab1), ("Emoji-Heavy", tab2), ("Edge Case", tab3)]:
            with tab:
                case_df = st.session_state.misclassified[
                    st.session_state.misclassified["Case Type"] == case_type
                    ]

                if len(case_df) > 0:
                    st.markdown(f"**{case_type} Cases - Common Errors**")

                    # Show most frequently misclassified examples
                    common_errors = case_df.groupby(
                        ["Text", "Expected", "Predicted"]
                    ).size().reset_index(name="Count").sort_values("Count", ascending=False)

                    st.dataframe(
                        common_errors.head(5),
                        column_config={
                            "Text": "Example Text",
                            "Expected": "True Label",
                            "Predicted": "Model Prediction",
                            "Count": "Error Frequency"
                        },
                        hide_index=True
                    )

                    # Show high-confidence errors
                    st.markdown("**High-Confidence Errors**")
                    st.dataframe(
                        case_df.nlargest(3, "Confidence")[
                            ["Model", "Text", "Expected", "Predicted", "Confidence"]
                        ],
                        hide_index=True
                    )
                else:
                    st.info(f"No misclassified {case_type.lower()} cases found")


# --- Main Execution ---
if st.button("▶️ Run Full Evaluation", type="primary"):
    run_evaluation()
    st.success("Evaluation completed successfully!")
    st.balloons()

if "results" in st.session_state:
    st.markdown("---")

    # Overall Metrics
    st.subheader("🏆 Overall Performance")
    col1, col2, col3 = st.columns(3)
    best_model = st.session_state.results.loc[st.session_state.results["Accuracy"].idxmax(), "Model"]
    col1.metric("Best Overall Model", best_model)
    col2.metric("Highest Accuracy", f"{st.session_state.results['Accuracy'].max():.1%}")
    col3.metric("Test Cases Evaluated", len(test_data))

    # Visualizations
    plot_category_performance()
    show_case_examples()

    # Misclassified examples
    st.subheader("❌ Misclassified Cases")
    model_filter = st.selectbox("Filter by Model:", ["All"] + list(MODELS.keys()))
    case_filter = st.selectbox("Filter by Case Type:", ["All", "Standard", "Sarcasm", "Emoji-Heavy", "Edge Case"])

    filtered = st.session_state.misclassified
    if model_filter != "All":
        filtered = filtered[filtered["Model"] == model_filter]
    if case_filter != "All":
        filtered = filtered[filtered["Case Type"] == case_filter]

    st.dataframe(
        filtered.sort_values("Confidence", ascending=False).head(20),
        column_config={
            "Model": "Model",
            "Text": "Text",
            "Expected": "True Label",
            "Predicted": "Prediction",
            "Confidence": st.column_config.NumberColumn(format="%.3f"),
            "Case Type": "Category"
        },
        hide_index=True,
        use_container_width=True
    )

    # Download options
    st.download_button(
        label="📥 Download Full Results",
        data=st.session_state.misclassified.to_csv(index=False).encode(),
        file_name="sentiment_analysis_results.csv",
        help="Includes all misclassified examples with model predictions"
    )

elif "results" not in st.session_state:
    st.info("Click 'Run Full Evaluation' to analyze model performance")