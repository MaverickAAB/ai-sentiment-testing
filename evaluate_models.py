import json
from turtle import st

from transformers import pipeline
from tqdm import tqdm  # Progress bar

#from streamlit_app import analyze_model_strengths

# Load test data
with open("test_data.json") as f:
    test_data = json.load(f)

# Models to evaluate
MODELS = {
    "Default": "distilbert-base-uncased-finetuned-sst-2-english",
    "Twitter": "cardiffnlp/twitter-roberta-base-sentiment",
    "3-Star": "finiteautomata/bertweet-base-sentiment-analysis"  # (Neutral support)
}

# Label mappings (adjust per model)
LABEL_MAPS = {
    "Default": {"POSITIVE": "positive", "NEGATIVE": "negative"},
    "Twitter": {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"},
    "3-Star": {"POS": "positive", "NEU": "neutral", "NEG": "negative"}
}

results = {}

for model_name, model_path in MODELS.items():
    print(f"\nEvaluating {model_name}...")
    pipe = pipeline("sentiment-analysis", model=model_path)
    correct = 0

    for item in tqdm(test_data):
        text = item["text"]
        expected = item["expected"]

        # Predict
        pred = pipe(text)[0]["label"]
        mapped_pred = LABEL_MAPS[model_name].get(pred, pred).lower()

        # Compare
        if mapped_pred == expected:
            correct += 1

    accuracy = correct / len(test_data)
    results[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy:.2%}")

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
#analyze_model_strengths(st.session_state.misclassified)