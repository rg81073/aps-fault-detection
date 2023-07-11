from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
import os,sys
from xgboost import XGBClassifier
from sensor import utils
from sklearn.metrics import f1_score
class ModelTrainer:

    
    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
            try:
                logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
                self.model_trainer_config=model_trainer_config
                self.data_transformation_artifact=data_transformation_artifact
      
            except Exception as e:
                raise SensorException(e, sys)
    
    def fine_tune(self):
        try:
            pass


        
        except Exception as e:
            raise SensorException(e, sys)



    def train_model(self,x,y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading Train and Test array.")

            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and Target Feature from both training and test array")

            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f" Train the Model")
            model =self.train_model(x=x_train,y=y_train)

            logging.info(f"Calculating f1_train_Score")

            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculating f1_test_Score")

            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)

            logging.info(f"train score: {f1_train_score} and test score: {f1_test_score}")

            # Check for Overfitting or Underfitting or Expected Score
            # For Underfitting and Expected Score--->

            logging.info(f"Checking if our Model is Underfitted or Not and Expected Score")

            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model Performance is not good as it not able to reach expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")

            # For Overfitting ---->

            logging.info(f"Checking if our model is Overfitting or Not")

            diff = abs(f1_train_score-f1_test_score)
            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test Score difference: {diff} is more than Overfitting threshold {self.model_trainer_config.overfitting_threshold}")
            
            # Saved the trained Model

            logging.info(f"Saving Trained model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # Preparing Artifact
            logging.info(f"Prepare the artifact")

            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)