from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import pandas as pd


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            City_Code=request.form.get('City_Code'),
            Accomodation_Type=request.form.get('Accomodation_Type'),
            Reco_Insurance_Type=request.form.get('Reco_Insurance_Type'),
            Is_Spouse=request.form.get('Is_Spouse'),
            Health_Indicator=request.form.get('Health_Indicator'),
            Holding_Policy_Duration=float(request.form.get('Holding_Policy_Duration')),
            Holding_Policy_Type=float(request.form.get('Holding_Policy_Type')),
            Reco_Policy_Cat=request.form.get('Reco_Policy_Cat'),
            Reco_Policy_Premium=request.form.get('Reco_Policy_Premium'),
            Age=request.form.get('Age'),

        )
        print(data)
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")  