import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

def load_test_data(file_path):
    # Load your test data (similar to how you loaded the training data)
    test_data = pd.read_csv(file_path)
    # Perform any necessary preprocessing on the test data

    return test_data

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Perform any necessary post-processing on the predictions

    # Example evaluation metrics for classification
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    # Example evaluation metrics for regression
    mse = mean_squared_error(y_test, y_pred)

    return accuracy, report, mse

if __name__ == "__main__":
    # Load the trained model
    model_path = "models/rnn_model.h5"
    model = load_model(model_path)

    # Load the test data
    test_data_path = "data/test_data.csv"
    test_data = load_test_data(test_data_path)

    # Perform the same preprocessing on the test data as you did for training data

    # Assuming X_test and y_test are the preprocessed test data and labels
    X_test, y_test = preprocess_test_data(test_data)

    # Evaluate the model
    accuracy, classification_report, mse = evaluate_model(model, X_test, y_test)

    # Print or log the evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report}")
    print(f"Mean Squared Error: {mse}")
