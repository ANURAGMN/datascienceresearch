from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Initialize variables for weights and training data
input_size = None
hidden_size = 3
output_size = None
hidden_weights = None
output_weights = None
X_train = None
y_train = None


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x, derivative=False) * (1 - sigmoid(x, derivative=False))
    else:
        return 1 / (1 + np.exp(-x))


def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    hidden_weights = np.random.uniform(-1, 1, (input_size, hidden_size))
    output_weights = np.random.uniform(-1, 1, (hidden_size, output_size))
    return hidden_weights, output_weights


def train(X, y, hidden_weights, output_weights, num_iterations=10000, alpha=0.1):
    for i in range(num_iterations):
        input_layer_outputs = X
        hidden_layer_inputs = np.dot(input_layer_outputs, hidden_weights)
        hidden_layer_outputs = sigmoid(hidden_layer_inputs)
        output_layer_inputs = np.dot(hidden_layer_outputs, output_weights)
        output_layer_outputs = sigmoid(output_layer_inputs)

        output_error = output_layer_outputs - y

        hidden_error = sigmoid(hidden_layer_outputs, derivative=True) * np.dot(output_error, output_weights.T)

        hidden_pd = np.dot(input_layer_outputs.T, hidden_error)
        output_pd = np.dot(hidden_layer_outputs.T, output_error)

        hidden_weights -= alpha * hidden_pd
        output_weights -= alpha * output_pd

    return hidden_weights, output_weights


def predict(X, hidden_weights, output_weights):
    hidden_layer_inputs = np.dot(X, hidden_weights)
    hidden_layer_outputs = sigmoid(hidden_layer_inputs)
    output_layer_inputs = np.dot(hidden_layer_outputs, output_weights)
    output_layer_outputs = sigmoid(output_layer_inputs)
    return output_layer_outputs, (output_layer_outputs >= 0.5).astype(int)


@app.route("/")
def index():
    return "backpropagation application"


@app.route("/train", methods=["POST"])
def train_model():
    global X_train, y_train, input_size, output_size, hidden_weights, output_weights

    # Retrieve training data from request
    data = request.json
    X_train = np.array(data.get('X_train'))
    y_train = np.array(data.get('y_train'))

    # Initialize weights based on input and output sizes
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_weights, output_weights = initialize_weights(input_size, hidden_size, output_size)

    # Train the model
    hidden_weights, output_weights = train(X_train, y_train, hidden_weights, output_weights)

    return jsonify({"message": "Model trained successfully!"})


# Endpoint to predict outputs for a test set
@app.route("/predict", methods=["POST"])
def predict_outputs():
    global hidden_weights, output_weights

    # Retrieve test data from request
    data = request.json
    X_test = np.array(data.get('X_test'))

    # Predict outputs using trained weights
    predicted_value, predicted_output = predict(X_test, hidden_weights, output_weights)

    # Prepare JSON response
    results = []
    for i in range(len(predicted_output)):
        results.append({
            "input": (X_test[i]).tolist(),  # Convert numpy int32 to Python int
            "predicted_output": float(predicted_output[i][0]),  # Convert numpy float64 to Python float
            "output_weight": float(predicted_value[i][0])  # Convert numpy float64 to Python float
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
'''
{
  "X_train": [
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1]
  ],
  "y_train": [
    [0],
    [1],
    [0],
    [1],
    [1],
    [0]
  ]
}

{
  "X_test": [
    [0, 1, 0]
  ]
}

'''