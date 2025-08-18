import json
import random
from faker import Faker

fake = Faker()

# Sentiment templates (expanded)
positive_phrases = [
    "I love {product}! It's amazing. 🔥", "Absolutely fantastic {product} 👏",
    "Best {product} ever! 💯", "Highly recommend {product}. ⭐⭐⭐⭐⭐",
    "{product} exceeded my expectations! 🎉", "Worth every penny! 💵",
    "Flawless victory with {product}! 🏆", "This {product} is a dream come true! 🌈"
]
neutral_phrases = [
    "The {product} was okay. 😐", "It's fine, nothing special. 🤷",
    "Average {product}. ⚖️", "Not bad, not great. 😑",
    "Meh, it's a {product}. 🫤", "Does the job, I guess. 🛠️",
    "{product} is... existent. 🌫️", "I have no strong feelings about {product}. 😐"
]
negative_phrases = [
    "I hate this {product}. 😡", "Terrible experience with {product}. 💀",
    "Worst {product} I've ever used. 🤮", "Avoid {product} at all costs! ☠️",
    "Total waste of money. 💸", "{product} broke after 2 days. 😠",
    "My cat could design a better {product}. 🐾", "{product} made me question reality. 🌀"
]

# New: Sarcasm templates (often misclassified)
sarcasm_phrases = [
    "Oh GREAT, another {product} that lasts a day. 🙄",
    "Wow, I LOVE spending $1000 on a paperweight. 👌",
    "{product} works PERFECTLY... if you enjoy frustration. 😒",
    "10/10 would NOT recommend {product}. 👍",
    "Nothing says 'quality' like {product} falling apart. 💅"
]

# New: Emoji-heavy templates
emoji_phrases = [
    "{product} = 🔥🔥🔥", "🚨 {product} ALERT: 🚀🚀🚀",
    "I’m 👏 here 👏 for 👏 this 👏 {product} 👏",
    "{product}? More like 💩", "🤯🤯🤯 {product} blew my mind!",
    "👎👎👎 {product} 👎👎👎", "🎉🎊 {product} saved my life! 🎊🎉"
]

# New: Edge cases (ambiguous/spam)
edge_case_phrases = [
    "The", "12345", "@#$%!", "👍", " ",
    "This sentence is false.", "I’m not sure how I feel about {product}.",
    "{product} is {product}.", "Yes. No. Maybe."
]

products = ["phone", "laptop", "movie", "restaurant", "book", "game", "service", "app", "car"]

# Generate 1500 test cases (1000 normal + 500 special cases)
test_data = []
for _ in range(1000):  # Standard cases
    product = random.choice(products)
    sentiment = random.choices(["positive", "neutral", "negative"], weights=[0.4, 0.2, 0.4])[0]

    if sentiment == "positive":
        text = random.choice(positive_phrases).format(product=product)
    elif sentiment == "neutral":
        text = random.choice(neutral_phrases).format(product=product)
    else:
        text = random.choice(negative_phrases).format(product=product)

    test_data.append({"text": text, "expected": sentiment})

# Add 500 special cases
for _ in range(200):  # Sarcasm (label as negative)
    product = random.choice(products)
    text = random.choice(sarcasm_phrases).format(product=product)
    test_data.append({"text": text, "expected": "negative"})  # Sarcasm is usually negative

for _ in range(200):  # Emoji-heavy
    product = random.choice(products)
    sentiment = random.choices(["positive", "negative"], weights=[0.5, 0.5])[0]
    if sentiment == "positive":
        text = random.choice([p for p in emoji_phrases if "🔥" in p or "🎉" in p])
    else:
        text = random.choice([p for p in emoji_phrases if "💩" in p or "👎" in p])
    test_data.append({"text": text, "expected": sentiment})

for _ in range(100):  # Edge cases (label as neutral)
    text = random.choice(edge_case_phrases)
    if "{product}" in text:
        text = text.format(product=random.choice(products))
    test_data.append({"text": text, "expected": "neutral"})

# Save to JSON
with open("test_data.json", "w") as f:
    json.dump(test_data, f, indent=2)

print("Generated test_data.json with 1500 samples (1000 standard + 500 edge cases).")