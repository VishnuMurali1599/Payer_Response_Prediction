import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object,load_object_deep
import os
from tensorflow.keras.models import  load_model


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = 'artifacts\proprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            preds = (preds > 0.5).astype(int)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        City_Code: str,
        Accomodation_Type: str,
        Reco_Insurance_Type:str,
        Is_Spouse: str,
        Health_Indicator: str,
        Holding_Policy_Duration: float,
        Holding_Policy_Type: float,
        Reco_Policy_Cat: int,
        Reco_Policy_Premium: float,
        Age: int
        ):

        self.City_Code = City_Code

        self.Accomodation_Type = Accomodation_Type

        self.Reco_Insurance_Type = Reco_Insurance_Type

        self.Is_Spouse = Is_Spouse

        self.Health_Indicator = Health_Indicator
        
        self.Holding_Policy_Duration = Holding_Policy_Duration

        self.Holding_Policy_Type = Holding_Policy_Type

        self.Reco_Policy_Cat = Reco_Policy_Cat
        
        self.Reco_Policy_Premium = Reco_Policy_Premium

        self.Age = Age


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "City_Code": [self.City_Code],
                "Accomodation_Type": [self.Accomodation_Type],
                "Reco_Insurance_Type": [self.Reco_Insurance_Type],
                "Is_Spouse": [self.Is_Spouse],
                "Health Indicator": [self.Health_Indicator],
                "Holding_Policy_Duration": [self.Holding_Policy_Duration],
                "Holding_Policy_Type": [self.Holding_Policy_Type],
                "Reco_Policy_Cat": [self.Reco_Policy_Cat],
                "Reco_Policy_Premium": [self.Reco_Policy_Premium],
                "Age": [self.Age]
            }

            #return pd.DataFrame(custom_data_input_dict)
            #return pd.DataFrame(pd.DataFrame.from_dict(custom_data_input_dict, orient='index').T)
            return pd.DataFrame(pd.DataFrame.from_dict(custom_data_input_dict, orient='index').T)

        except Exception as e:
            raise CustomException(e, sys)