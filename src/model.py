from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

def build_rnn_model(input_shape, vocab_size, embedding_dim, rnn_units, output_dim):
    model = Sequential()

    # Embedding layer to convert integer indices to dense vectors
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_shape[1]))

    # SimpleRNN layer with a specified number of units
    model.add(SimpleRNN(units=rnn_units, activation='relu'))

    # Dense output layer for classification or regression
    model.add(Dense(units=output_dim, activation='softmax'))  # Adjust activation based on your task

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # Example usage
    vocab_size = 10000  # Choose an appropriate vocabulary size
    embedding_dim = 50  # Choose an appropriate dimension for word embeddings
    rnn_units = 50  # Choose the number of units in the SimpleRNN layer
    output_dim = 10  # Adjust based on the number of classes for classification or 1 for regression

    input_shape = (max_sequence_length,)  # Adjust based on your sequence length

    rnn_model = build_rnn_model(input_shape, vocab_size, embedding_dim, rnn_units, output_dim)

    # Display the summary of the model architecture
    rnn_model.summary()

    # Save the model to a file for later use in training and evaluation
    rnn_model.save("models/rnn_model.h5")
