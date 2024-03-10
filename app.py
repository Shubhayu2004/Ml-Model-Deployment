from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load any necessary preprocessing objects (e.g., scalers)
# scaler = joblib.load('your_scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form submission
    house_number = request.form['house_number']
    date = request.form['date']

    # Perform feature engineering
    # For example, extract day of the week, month, and season from the date
    # Example code for feature engineering:
    date_features = pd.to_datetime(date)
    day_of_week = date_features.dayofweek
    month = date_features.month
    season = (date_features.month % 12 + 3) // 3  # Divide months into seasons (1: Winter, 2: Spring, 3: Summer, 4: Autumn)

    # Create DataFrame with input features
    input_df = pd.DataFrame({'house_number': [house_number],
                             'day_of_week': [day_of_week],
                             'month': [month],
                             'season': [season]})

    # Make predictions
    predictions = model.predict(input_df)

    # Format prediction for display
    prediction_text = f'Predicted Energy Demand: {predictions[0]}'

    # Render the template with prediction
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
