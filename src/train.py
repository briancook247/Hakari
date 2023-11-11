from keras.callbacks import ModelCheckpoint
from model import build_rnn_model
import numpy as np
from preprocessing import load_data, preprocess_text_data, preprocess_time_series_data, split_data, standardize_data, load_test_data
from evaluate import evaluate_model

def train_rnn_model(X_train, y_train, model, batch_size, epochs, validation_data=None, checkpoint_path=None):
    # Optionally, use ModelCheckpoint to save the best weights during training
    callbacks = []
    if checkpoint_path:
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True)
        callbacks.append(checkpoint_callback)

    # Train the model
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=validation_data, callbacks=callbacks)

    return history

if __name__ == "__main__":
    # Example usage
    file_path_train = "data/train_data.csv"
    data_train = load_data(file_path_train)

    # Assuming 'text_data' is a column in the CSV file containing text sequences
    texts_train = data_train['text_data'].values

    max_words = 10000  # Choose an appropriate vocabulary size
    max_sequence_length = 50  # Choose an appropriate sequence length

    X_train, y_train = preprocess_text_data(texts_train, max_words, max_sequence_length)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(X_train, y_train, test_size=0.2, random_state=42)

    # Optionally, standardize the data (if needed)
    # X_train_std, X_val_std = standardize_data(X_train, X_val)

    # Build the RNN model
    input_shape = (max_sequence_length,)  # Adjust based on your sequence length
    vocab_size = 10000  # Should match the vocab_size used in preprocessing
    embedding_dim = 50  # Should match the embedding_dim used in model.py
    rnn_units = 50  # Should match the rnn_units used in model.py
    output_dim = 10  # Adjust based on the number of classes for classification or 1 for regression

    rnn_model = build_rnn_model(input_shape, vocab_size, embedding_dim, rnn_units, output_dim)

    # Train the RNN model
    batch_size = 32
    epochs = 10
    checkpoint_path = "models/rnn_model_best.h5"  # Optional: specify a path to save the best weights
    history = train_rnn_model(X_train, y_train, rnn_model, batch_size, epochs,
                              validation_data=(X_val, y_val), checkpoint_path=checkpoint_path)

    # Optionally, save the entire model after training
    rnn_model.save("models/rnn_model_final.h5")

    # Now, load and preprocess the test data
    file_path_test = "data/test_data.csv"
    data_test = load_test_data(file_path_test)

    # Assuming 'text_data' is a column in the CSV file containing text sequences
    texts_test = data_test['text_data'].values

    X_test, y_test = preprocess_text_data(texts_test, max_words, max_sequence_length)

    # Evaluate the model on the test data
    accuracy, classification_report, mse = evaluate_model(rnn_model, X_test, y_test)

    # Print or log the evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report}")
    print(f"Mean Squared Error: {mse}")
