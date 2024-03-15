import os
import sys

import numpy as np 
import pandas as pd
#import dill
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import  load_model

from src.exception import CustomException

def save_object(file_path, obj):  # Saving Preprocessing FIle
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def save_object_deep(file_path, obj):  ## Saving Deep Learning File
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object_deep(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return load_model(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    