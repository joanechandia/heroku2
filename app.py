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
    pr = [float(x) for x in request.form.values()]
    final_features=[[pr[0],pr[1],pr[2],pr[0]-pr[1],pr[0]+pr[1],pr[0]*pr[1],pr[1]/(pr[0]/100)**2,pr[1]/pr[0]]]
    prediction = model.predict(final_features)

    output =prediction[0]

    return render_template('index.html',
    
     prediction_text='0 - Extremadamente débil; 1 - Débil; 2 - Normal; 3 - Sobrepeso;  4 - Obesidad; 5 - Obesidad extrema. Su peso corporal es:  {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)