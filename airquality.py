from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import csv
import os
from datetime import datetime
import pandas as pd

app = Flask(__name__)

PREDICTIONS_FOLDER = 'predictions'
PREDICTIONS_FILE = os.path.join(PREDICTIONS_FOLDER, 'predictions.csv')

# Ensure the predictions folder exists
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Load the trained model
model_path = 'C:\\Users\\rushabh nakum\\Desktop\\Minor Project 7 sem\\pickel_file\\model_pickel_random_forest.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to initialize the CSV file if it doesn't exist
def init_csv():
    if not os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'input_data', 'prediction'])

init_csv()

# Function to append a new prediction to the CSV
def append_prediction(timestamp, input_data, prediction):
    with open(PREDICTIONS_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, str(input_data), prediction])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [float(x) for x in request.form.values() if x.isnumeric() or x.replace('.', '', 1).isdigit()]
    final_features = [np.array(int_features)]
    
    # Print input for debugging
    print("Input features:", int_features)
    print("Shaped features:", final_features)
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Print raw prediction for debugging
    print("Raw model prediction:", prediction)
    
    # Map prediction to categories
    output_map = {
        'Good' :1,
        'Satisfactory' :2,
        'Moderate' :3,
        'Poor' :4,
        'Very Poor' :5,
        'Severe' :6
    }
    
    output = output_map.get(prediction[0], "Unknown Category")
    
    # Print mapped prediction for debugging
    print("Mapped prediction:", output)
    
    # Append prediction to CSV
    timestamp = datetime.now().isoformat()
    append_prediction(timestamp, int_features, output)
    
    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

# Function to get the latest and previous predictions
def get_latest_and_previous():
    df = pd.read_csv(PREDICTIONS_FILE)
    if len(df) < 2:
        return None, None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=False)
    
    latest = df.iloc[0]
    previous = df.iloc[1]
    
    return latest, previous

# API endpoint to compare predictions
@app.route('/compare', methods=['GET'])
def compare_predictions():
    latest, previous = get_latest_and_previous()
    
    if latest is None or previous is None:
        return jsonify({'error': 'Not enough predictions to compare'})
    
    return jsonify({
        'latest_prediction': latest['prediction'],
        'previous_prediction': previous['prediction'],
        'latest_input': latest['input_data'],
        'previous_input': previous['input_data']
    })

if __name__ == "__main__":
    app.run(debug=True)
    