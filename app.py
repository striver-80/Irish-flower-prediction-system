from flask import Flask, render_template, request
import numpy as np
import pickle as pkl
app = Flask(__name__) 

model = pkl.load(open('model.pkl', 'rb'))
@app.route('/') 
def home(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST']) 
def predict(): 
    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']
    features = [[sepal_length, sepal_width, petal_length, petal_width] ]
    predicted_score = model.predict(features)
    return render_template('predict.html', predictions=predicted_score) 
if __name__ == '__main__': 
    app.run(debug=True) 