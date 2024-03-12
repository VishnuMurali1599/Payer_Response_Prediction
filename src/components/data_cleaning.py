import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#from src.components.data_transforamtion import DataTransformation
#from src.components.data_transforamtion import DataTransformationConfig

#from src.components.model_trainer import ModelTrainerConfig
#from src.components.model_trainer import ModelTrainer

@dataclass
class DataCleaningConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")

class DataCleaning:
    def __init__(self):
        self.ingestion_config=DataCleaningConfig()

    def initiate_data_cleaning(self):
        logging.info("Entered the data Cleaning method")
        try:
            df=pd.read_csv('artifacts\data.csv')
            
            logging.info("Raw data is ingested and Proceeding to data cleaning")
            df['Age'] = df.apply(lambda row: row['Upper_Age'] if row['Upper_Age'] == row['Upper_Age'] else (row['Upper_Age'] + row['Upper_Age']) / 2, axis=1)
            df = df.drop(columns=['Upper_Age','Lower_Age'],axis=1)
            logging.info("final dataframe is obtained before data cleaning")
            
            ## Filling Health indicator with Mode value
            logging.info("Cleaning Health indicator column")
            most_common_category = df['Health Indicator'].mode().iloc[0]
            df['Health Indicator'] = df['Health Indicator'].fillna(most_common_category)
            
            logging.info("replacing policy duration values and filling missing values")
            df['Holding_Policy_Duration'] = pd.to_numeric(df['Holding_Policy_Duration'].replace('14+', '15'))
            df['Holding_Policy_Duration'] = df['Holding_Policy_Duration'].fillna(df['Holding_Policy_Duration'].median())
            
            logging.info("Filling Policy type missing values")
            df['Holding_Policy_Type'] = df['Holding_Policy_Type'].fillna(df['Holding_Policy_Type'].median())

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
