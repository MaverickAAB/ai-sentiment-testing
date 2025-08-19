# Load and run your sentiment analysis model here
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment(text):
    """Predict sentiment of text using Hugging Face Transformers."""
    result = sentiment_pipeline(text)[0]
    return {
        "label": result['label'],
        "score": round(result['score'], 4)
    }

if __name__ == "__main__":
    text_input = "I absolutely loved the movie!"
    prediction = get_sentiment(text_input)
    print(f"Sentiment: {prediction['label']} (confidence: {prediction['score']})")

""" 
Why This Model? Hugging Face chooses distilbert-base-uncased-finetuned-sst-2-english as the default when you call pipeline("sentiment-analysis")
# It's trained on SST-2 (Stanford Sentiment Treebank v2), a gold standard dataset for sentiment analysis
from transformers import pipeline
# This is from the Hugging Face Transformers library.
# Load the sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
# This is a high-level utility that lets you load pre-trained models easily without writing model/tokenizer code manually.
# This line automatically:#
# Downloads a default pre-trained model (distilbert-base-uncased-finetuned-sst-2-english) from Hugging Face
# Loads its tokenizer and model#
# Creates a pipeline that can take raw text input and return sentiment output
# Equals to this line of code -from transformers import AutoTokenizer, AutoModelForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
def get_sentiment(text):
    
    Predict sentiment of given text using Hugging Face Transformers.
    Returns a dictionary: {'label': 'POSITIVE', 'score': 0.99}
    
    result = sentiment_pipeline(text)[0]
        return {
           "label": result['label'],
           "score": round(result['score'], 4)
    }
# text is passed to the pipeline. # The model tokenizes it, processes it through BERT, and applies a classification head to assign a label
# # The [0] just grabs the first result (you can input multiple texts)
# label: The predicted class#
# score: The softmax confidence score for that class (ranges from 0 to 1)
# round(result['score'], 4)#
# We round the confidence score to 4 decimal places for cleaner output
# Example
if __name__ == "__main__":
 text_input = "I absolutely loved the movie!"
  prediction = get_sentiment(text_input)
   print(f"Sentiment: {prediction['label']} (confidence: {prediction['score']})")

"""