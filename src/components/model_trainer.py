import os
import sys
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object_deep


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = Sequential()
            models.add(Dense(8,activation="relu",input_dim=10))
            models.add(Dense(6,activation="relu"))
            models.add(Dense(1,activation='sigmoid'))
            
            models.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            models.fit(X_train,y_train,epochs=20, batch_size=32,validation_split=0.2,verbose=0)
            
            y_train_pred = models.predict(X_train)
            y_train_pred = (y_train_pred > 0.5).astype(int)

            y_test_pred = models.predict(X_test)
            y_test_pred = (y_test_pred > 0.5).astype(int)
            

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_accuracy = accuracy_score(y_train,y_train_pred)
            
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            
            models.save("OP.h5")
            print("one save is done")
            
            save_object_deep(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models.save(self.model_trainer_config.trained_model_file_path)
            )
            
            print("Another save is also done")
            
            return (train_model_score,test_model_score)
            
        except Exception as e:
            raise CustomException(e,sys)