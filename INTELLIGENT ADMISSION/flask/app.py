from flask import Flask, request, render_template, jsonify
from tensorflow import keras
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)
# Import necessary libraries

# model = pickle.load(open('university.pkl', 'rb))
# load model trained model
# Load your trained model


model = keras.models.load_model('C:/Users/ELCOT/PycharmProjects/pythonProject/flask/model.h5')
scaler=joblib.load('C:/Users/ELCOT/PycharmProjects/pythonProject/flask/scaler.pkl')

@app.route('/')
def entry():
    return render_template('home.html')


@app.route('/getdata', methods=['post'])
def home():
    gre = request.form['gre']
    toefl = request.form['toefl']
    uni_no = request.form['uni_num']
    sop = request.form['sop']
    lor = request.form['lor']
    cgpa = request.form['cgpa']
    research = request.form['Research']
    variables = [[int(gre), int(toefl), int(uni_no), float(sop), float(lor), float(cgpa), int(research)]]
    # Define new input data
    # new_data = [[298, 98, 2, 4.0, 3.0, 8.03, 0]]

    # Scale the new input data using the same scaler object
    scaled_data = scaler.transform(variables)

    # Make a prediction using the scaled data
    # prediction = lr.predict(scaled_data)

    # Print the prediction
    #  print(prediction)

    result = model.predict(scaled_data)
    if result>=0.5:
        result_f=' get admission in university'
    else:
        result_f=' not get admission in university'
    return render_template('no_change.html', output=result_f)


if __name__ == "__main__":
    app.run(debug=True)
