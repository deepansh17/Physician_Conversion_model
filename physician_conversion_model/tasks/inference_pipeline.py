#Basic ML dependencies
import pandas as pd
import numpy as np

from io import BytesIO
import datetime
import hopsworks
import importlib.util
import sys
import yaml
import mlflow

class Inferencemodel():
    def __init__(self, conf):
        self.conf = conf
        self.bucket_name = self.conf['s3']['bucket_name']
        self.aws_region = self.conf['s3']['aws_region']
        self.inference_path = self.conf['preprocessed']['inference_df_path']
        self.id_drop_column_list = self.conf['feature_transformation']['id_col_list']


    def load_module(self, file_name, module_name):
        spec = importlib.util.spec_from_file_location(module_name, file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    def model_inference(self):
        my_module = self.load_module("./physician_conversion_model/tasks/utils.py", "utils")
        utils_func = my_module.utils()
        df_input = utils_func.load_data_from_s3(self.bucket_name, self.aws_region, self.inference_path)
        drop_list = self.id_drop_column_list.append("TARGET")
        df_inference = df_input.drop(drop_list, axis=1)
        model = mlflow.load_model(registered_model_name="xgboost-model")
        
if __name__ == '__main__':
    with open('./conf/tasks/feature_pipepline.yml', 'r') as config_file:
        configuration = yaml.safe_load(config_file)
    task = Inferencemodel(configuration)
    task.model_inference()