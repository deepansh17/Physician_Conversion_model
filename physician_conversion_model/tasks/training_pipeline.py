#Basic ML dependencies
import pandas as pd
import numpy as np

#ML Training dependencies
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Importing necessary libraries for model development and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import xgboost as xgb
from urllib.parse import urlparse
import mlflow
from mlflow.tracking.client import MlflowClient

# # Hyperparameter Tuning
#from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

#System and Env Dependencies
import warnings
import os
import boto3
import urllib
import pickle
from io import BytesIO
import datetime
import hopsworks
import importlib.util
import sys
import yaml

#warnings
warnings.filterwarnings('ignore')


class Trainmodel():
    def __init__(self, conf):
        self.conf = conf
        self.api_key = self.conf['hopsworks_feature_store']['api_key']
        self.project_name = self.conf['hopsworks_feature_store']['project_name']
        self.bucket_name = self.conf['s3']['bucket_name']
        self.aws_region = self.conf['s3']['aws_region']
        self.input_training_path = self.conf['preprocessed']['train_df_path']
        self.col_list = self.conf['hopsworks_feature_store']['lookup_key']
        self.table_name = self.conf['hopsworks_feature_store']['table_name']
        self.folder_path = self.conf['preprocessed']['model_variable_list_file_path']
        self.file_name = self.conf['preprocessed']['model_variable_list_file_name']
        self.feature_view = self.conf['hopsworks_feature_store']['feature_view']
        self.id_drop_column_list = self.conf['feature_transformation']['id_col_list']
        self.best_params = self.conf['train_model_parameters']['params']
        
    def load_module(self, file_name, module_name):
        spec = importlib.util.spec_from_file_location(module_name, file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    def model_train(self):
        project = hopsworks.login(
            api_key_value=self.api_key,
            project=self.project_name,
        )
        my_module = self.load_module("./physician_conversion_model/tasks/utils.py", "utils")
        utils_func = my_module.utils()
        df_input = utils_func.load_data_from_s3(self.bucket_name, self.aws_region, self.input_training_path)
        

        df_input = df_input.reset_index()
        df_input.drop(['index'], axis = 1, inplace = True, errors= 'ignore')

        #Clean column names
        df_input.columns = df_input.columns.str.strip()
        df_input.columns = df_input.columns.str.replace(' ', '_')

         #Convert ID columns to string type
        
        utils_func.convert_columns_to_string(df_input, self.col_list)
        # Get features from Hopsworks
        fs = project.get_feature_store()
        features_df = fs.get_feature_group(self.table_name , version=1)
        file_select_features = self.folder_path+self.file_name
        
        #load data from pickle file
        model_features_list = utils_func.load_pickle_from_s3(self.bucket_name,self.aws_region, file_select_features)
        query = features_df.select(model_features_list)
        
        #create feature view
        feature_view = fs.get_or_create_feature_view(
        name=self.feature_view,
        version=4,
        query=query,
        labels=["target"]
        )
        TEST_SIZE = 0.2
        td_version, td_job = feature_view.create_train_test_split(
            description = 'Physician Conversion model train feature data',
            data_format = 'csv',
            test_size = TEST_SIZE
        )
        
        # create a training dataset as DataFrame
        X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=TEST_SIZE)
        with mlflow.start_run():
            drop_id_col_list = self.id_drop_column_list
            #mlflow.log_params(self.best_params)
            
            # Train the final model with the best hyperparameters
            best_model = xgb.XGBClassifier(**self.best_params, random_state=321)
            best_model.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)
            
            # Evaluate the final model on a test dataset (X_test, y_test)
            test_score = best_model.score(X_test.drop(drop_id_col_list, axis=1, errors='ignore'), y_test)
            
            # Log evaluation metric (e.g., accuracy)
            mlflow.log_metric("test_accuracy", test_score)
            
            # Log the trained model using MLflow's XGBoost log function
            mlflow.xgboost.log_model(best_model,artifact_path="usecase", registered_model_name="xgboost-model")
            
            #log confusion metrics
            utils_func.eval_cm(best_model, X_train, y_train, X_test,
                                            y_test,drop_id_col_list)
            
            # log roc curve
            utils_func.roc_curve(best_model, 
                            X_test,y_test,drop_id_col_list)
            
            #Log model evaluation metrics
            mlflow.log_metrics(utils_func.evaluation_metrics(
                best_model,
                X_train, y_train, 
                X_test, y_test,
                  drop_id_col_list))
            

            mlflow.log_artifact('confusion_matrix_train.png')
            mlflow.log_artifact('confusion_matrix_validation.png')
            mlflow.log_artifact('roc_curve.png')
            
        
        
if __name__ == '__main__':
    with open('./conf/tasks/feature_pipepline.yml', 'r') as config_file:
        configuration = yaml.safe_load(config_file)
    task = Trainmodel(configuration)
    task.model_train()