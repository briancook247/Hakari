import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    # Load data from a CSV file or another format
    data = pd.read_csv(file_path)
    return data

def preprocess_text_data(texts, max_words, max_sequence_length):
    # Tokenize and pad text data
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    return padded_sequences, tokenizer.word_index

def preprocess_time_series_data(time_series_data, sequence_length):
    # Preprocess time series data (assuming it's a single feature)
    X, y = [], []
    for i in range(len(time_series_data) - sequence_length):
        X.append(time_series_data[i:i+sequence_length])
        y.append(time_series_data[i+sequence_length])
    
    X, y = np.array(X), np.array(y)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    # Standardize the data (if needed)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

# Example usage:
if __name__ == "__main__":
    file_path = "data/train_data.csv"
    data = load_data(file_path)

    # Assuming 'text_data' is a column in the CSV file containing text sequences
    texts = data['text_data'].values

    max_words = 10000  # Choose an appropriate vocabulary size
    max_sequence_length = 50  # Choose an appropriate sequence length

    padded_sequences, word_index = preprocess_text_data(texts, max_words, max_sequence_length)
    
    # Assuming 'time_series_data' is a column in the CSV file containing time series data
    time_series_data = data['time_series_data'].values
    
    sequence_length = 10  # Choose an appropriate sequence length for time series data
    X, y = preprocess_time_series_data(time_series_data, sequence_length)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Standardize the data if needed
    X_train_std, X_test_std = standardize_data(X_train, X_test)

    # You can then save these preprocessed data for later use in training your RNN
    # Save padded_sequences, word_index, X_train, X_test, y_train, y_test, X_train_std, X_test_std to appropriate files
