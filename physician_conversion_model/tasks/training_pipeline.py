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
# import mlflow
# from mlflow.tracking.client import MlflowClient

# # Hyperparameter Tuning
# from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

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
        self.input_training_path = self.conf['preprocessed']['preprocessed_df_path']
        self.col_list = self.conf['hopsworks_feature_store']['lookup_key']
        self.table_name = self.conf['hopsworks_feature_store']['table_name']
        self.folder_path = self.conf['preprocessed']['model_variable_list_file_path']
        self.file_name = self.conf['preprocessed']['model_variable_list_file_name']
        self.feature_view = self.conf['hopsworks_feature_store']['feature_view']
        
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
        version=1,
        query=query,
        labels=["target"]
        )
        TEST_SIZE = 0.2
        VALIDATION_SIZE = 0.3
        td_version, td_job = feature_view.create_train_test_split(
            description = 'Physician Conversion model feature data',
            data_format = 'csv',
            validation_size = VALIDATION_SIZE,
            test_size = TEST_SIZE
        )
        
        # create a training dataset as DataFrame
        X_train, X_val, X_test, y_train, y_val, y_test = feature_view.train_validation_test_split(validation_size=VALIDATION_SIZE, test_size=TEST_SIZE,random_state=42)
       
        
        
if __name__ == '__main__':
    with open('./conf/tasks/feature_pipepline.yml', 'r') as config_file:
        configuration = yaml.safe_load(config_file)
    task = Trainmodel(configuration)
    task.model_train()