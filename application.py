from flask import Flask,request,render_template,jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
application=Flask(__name__)
app=application
logistic_model=pickle.load(open(r'C:\Users\ys136\Desktop\Data Science\End to End ML Projects\Diabetes Prediction\Models\logisticregression.pkl','rb'))

Standard_scaler=pickle.load(open(r'C:\Users\ys136\Desktop\Data Science\End to End ML Projects\Diabetes Prediction\Models\standardscaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict_datapoint',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))

        newdata_scaled=Standard_scaler.transform([[Pregnancies,	Glucose,	BloodPressure,	SkinThickness,	Insulin,	BMI	,DiabetesPedigreeFunction,	Age	]])
        Predict=logistic_model.predict(newdata_scaled)
        if Predict[0]==1:
            result='Diabetic'
            
        else:
            result='Not Diabetic'

        return render_template('single_prediction.html',result=result)
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5501,debug=True)


