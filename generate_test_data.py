import json
import random
from faker import Faker

fake = Faker()

# Sentiment templates (weighted for balance)
positive_phrases = [
    "I love {product}! It's amazing.", "Absolutely fantastic {product}.",
    "Best {product} ever!", "Highly recommend {product}. ⭐⭐⭐⭐⭐",
    "{product} exceeded my expectations!", "Worth every penny!"
]
neutral_phrases = [
    "The {product} was okay.", "It's fine, nothing special.",
    "Average {product}.", "Not bad, not great.",
    "Meh, it's a {product}.", "Does the job, I guess."
]
negative_phrases = [
    "I hate this {product}.", "Terrible experience with {product}.",
    "Worst {product} I've ever used.", "Avoid {product} at all costs!",
    "Total waste of money.", "{product} broke after 2 days. 😠"
]

products = ["phone", "laptop", "movie", "restaurant", "book", "game", "service"]

# Generate 1000 test cases
test_data = []
for _ in range(1000):
    product = random.choice(products)
    sentiment = random.choices(["positive", "neutral", "negative"], weights=[0.4, 0.2, 0.4])[0]

    if sentiment == "positive":
        text = random.choice(positive_phrases).format(product=product)
    elif sentiment == "neutral":
        text = random.choice(neutral_phrases).format(product=product)
    else:
        text = random.choice(negative_phrases).format(product=product)

    test_data.append({"text": text, "expected": sentiment})

# Save to JSON
with open("test_data.json", "w") as f:
    json.dump(test_data, f, indent=2)

print("Generated test_data.json with 1000 samples.")