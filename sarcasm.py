
# from google.colab import files
# uploaded = files.upload()
import nltk
nltk.download('vader_lexicon')
import tensorflow as tf
import numpy as np
import json
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from colorama import Fore, Style

# Download necessary nltk resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load sarcasm dataset (Ensure you have sarcasm dataset JSON file)
with open('Sarcasm_Headlines_Dataset.json', 'r') as f:
    data = json.load(f)

sentences = []
labels = []
for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# Tokenization
max_words = 5000
max_length = 20
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(padded, labels, np.array(labels),test_size=0.2, random_state=42)

# Model for sarcasm detection
model = Sequential([
    Embedding(max_words, 32, input_length=max_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, np.array(y_train), epochs=5, validation_data=(X_test, np.array(y_test)), batch_size=32)

# Function to predict sentiment and sarcasm
def analyze_text(text):
    # Sentiment Analysis
    sentiment_score = sia.polarity_scores(text)['compound']
    if sentiment_score >= 0.05:
        sentiment = "Positive"
        color = Fore.GREEN
    elif sentiment_score <= -0.05:
        sentiment = "Negative"
        color = Fore.RED
    else:
        sentiment = "Neutral"
        color = Fore.YELLOW
    
    # Sarcasm Detection
    sequence = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(sequence, maxlen=max_length, padding='post')
    sarcasm_pred = model.predict(padded_seq, verbose=0)[0][0]
    sarcasm = "Sarcastic" if sarcasm_pred > 0.5 else "Not Sarcastic"
    
    print(f"{color}Text: {text}\nSentiment: {sentiment} \nSarcasm: {sarcasm}{Style.RESET_ALL}\n")

# Test the function
sample_texts = [
    "Wow, this is the best day of my life!",
    "I just love waiting in traffic for hours.",
    "The weather is great today!",
    "Oh, fantastic. Another Monday morning.",
    "Oh, fantastic! Another Monday morning. Just what I needed."
]

for text in sample_texts:
    analyze_text(text)