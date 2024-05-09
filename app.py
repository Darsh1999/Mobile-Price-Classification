from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('/Users/darshdave/Documents/Projects/Mobile-Price-Classification/MODEL/hyperparameter_tuned_logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction_text=f'Predicted Price Range: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)