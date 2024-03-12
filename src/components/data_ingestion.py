import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_cleaning import DataCleaning
from src.components.data_cleaning import DataCleaningConfig

#from src.components.data_transforamtion import DataTransformation
#from src.components.data_transforamtion import DataTransformationConfig

#from src.components.model_trainer import ModelTrainerConfig
#from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df=pd.read_csv('research/data/train.csv')
            df.drop(columns=["ID","Region_Code"],axis=1,inplace=True)
            
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
    
            logging.info("Ingestion of data is completed (Raw data)")

            return(
                self.ingestion_config.raw_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    raw_data=obj.initiate_data_ingestion()
    
    obj=DataCleaning()
    train_data,test_data=obj.initiate_data_cleaning()