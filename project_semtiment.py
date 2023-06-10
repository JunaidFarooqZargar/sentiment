import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Step 1: Prepare the dataset
# Replace `texts` and `labels` with your own dataset
texts = ["I love this movie!", "This is terrible.", "The acting was great."]
labels = [1, 0, 1]  # 1 represents positive sentiment, 0 represents negative sentiment

# Step 2: Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max([len(seq) for seq in sequences])
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Step 3: Prepare training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Step 4: Build the RNN model
model = Sequential()
model.add(Embedding(len(word_index) + 1, 128, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# Step 5: Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Step 6: Create a user-friendly interface for sentiment analysis
def analyze_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model.predict(sequence)[0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment

# Example usage
text_input = input("Enter a text: ")
result = analyze_sentiment(text_input)
print("Sentiment: ", result)