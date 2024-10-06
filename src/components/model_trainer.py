import sys
import os
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.utils import saved_obj
from dataclasses import dataclass
from src.utils import evaluate_model

@dataclass
class modelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class modeltrainer:
    def __init__(self) :
        self.model_trainer_config=modelTrainerConfig()

    def initiated_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split the train and test data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                #"linear regressor":LinearRegression(),
                "cat boost":CatBoostRegressor(),
                "xgboost":XGBRegressor(),
                "random forest":RandomForestRegressor(),
                "kneighbour":KNeighborsRegressor(),
                "ada boost":AdaBoostRegressor(),
                "gradient boost":GradientBoostingRegressor(),
                "decision tree":DecisionTreeRegressor()

            }
            model_report=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            
            
            logging.info("best model found")

            saved_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            score=r2_score(y_test,predicted)
            return score
        


        except Exception as e:
            raise CustomException(e,sys)
        

