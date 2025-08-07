from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)
model = pickle.load(open("svr_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['fixed_acidity']),
        float(request.form['volatile_acidity']),
        float(request.form['citric_acid']),
        float(request.form['alcohol']),
        float(request.form['pH']),
        
    ]
    prediction = model.predict([features])[0]
    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
