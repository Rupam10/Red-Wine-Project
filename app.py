import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Wine_EDA.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('indexx.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['fixed acidity','volatile acidity',
    'citric acid','residual sugar','chlorides',
    'free sulphur dioxide','total sulphur dioxide',
    'density','pH','sulphates','alcohol Output variable']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** Good Wine **"
    else:
        res_val = "Bad Wine"
        

    return render_template('indexx.html', prediction_text='Wine is {}'.format(res_val))

if __name__ == "__main__":
    app.run()
