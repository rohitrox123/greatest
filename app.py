import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = [str(request.form['comment'])]
    prediction = model.predict([final_features])

    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted Rate for your comment: $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)