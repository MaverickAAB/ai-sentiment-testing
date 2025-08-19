# Sentiment Analysis Model Comparison

## Features
- Compare Default/Twitter/3-Star models
- Specialized analysis for:
  - 😜 Sarcasm
  - 🔥 Emoji-heavy text
  - ✂️ Edge cases

## Quick Start
```bash
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run streamlit_app.py
```
## Data Format
`test_data.json` example:
```json
{"text":"This is great!","expected":"positive"}
```