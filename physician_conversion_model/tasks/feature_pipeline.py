import pandas as pd
import numpy as np
import warnings
import yaml
import os
import boto3
import importlib.util
import sys

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

from io import BytesIO

warnings.filterwarnings('ignore')

class DataPrep:

    def __init__(self, conf):
        self.conf = conf
        self.bucket_name = self.conf['s3']['bucket_name']
        self.aws_region = self.conf['s3']['aws_region']
        self.file_path = self.conf['s3']['file_path']
        self.drop_col_list = self.conf['feature_transformation']['drop_column_list']
        self.encode_col_list = self.conf['feature_transformation']['one_hot_encode_feature_list']
        self.threshold = self.conf['param_values']['variance_threshold_value']
        self.n = self.conf['param_values']['select_k_best_feature_num']
        self.id_col_list = self.conf['feature_transformation']['id_col_list']
        self.target_col = self.conf['feature_transformation']['target_col']
        self.folder_path = self.conf['preprocessed']['model_variable_list_file_path']
        self.file_name = self.conf['preprocessed']['model_variable_list_file_name']
        self.s3_object_key = self.conf['preprocessed']['preprocessed_df_path']

    def load_module(self, file_name, module_name):
        spec = importlib.util.spec_from_file_location(module_name, file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


    def preprocess_data(self):
        """
        Perform data preprocessing steps as specified in the configuration.

        This method loads data from an S3 bucket, cleans the data, performs one-hot encoding,
        applies feature selection, and saves the preprocessed data back to S3.
        """

        my_module = self.load_module("./physician_conversion_model/tasks/utils.py", "utils")
        utils_func = my_module.utils()

        df_input = utils_func.load_data_from_s3(self.bucket_name, self.aws_region, self.file_path)
        
        #Clean column names
        df_input.columns = df_input.columns.str.strip()
        df_input.columns = df_input.columns.str.replace(' ', '_')
        
        #Drop unwanted column: "HCO Affiliation" - "Affiliation Type" is more valid column for us
        df_input.drop(self.drop_col_list, axis=1, inplace=True)

        #One hot encode categorical features
        df_input = pd.get_dummies(df_input, columns=self.encode_col_list, drop_first=True)
        
        #Select variables for feature selection
        id_target_col_list = self.conf['feature_transformation']['id_target_col_list']
        col_for_feature_selection = df_input.columns.difference(id_target_col_list)
        
        #Variance threshold feature selection method
        var_thr = VarianceThreshold(threshold=self.threshold)
        var_thr.fit(df_input[col_for_feature_selection])

        df_input_subset = df_input[col_for_feature_selection]
        remove_col_list = [col for col in df_input_subset.columns if col not in df_input_subset.columns[var_thr.get_support()]]
        
        #remove above list column from master dataframe
        df_input.drop(remove_col_list, axis=1, inplace=True, errors='ignore')

        #Feature Selection Using Select K Best
        df = df_input.drop(self.id_col_list, axis=1)
        target_col_var = df_input[self.target_col]
        top_n_col_list = utils_func.select_kbest_features(df, target_col_var, self.n)
        
        #Convert to list
        top_n_col_list = top_n_col_list.tolist()

        # Dump top_n_col_list to s3 bucket
        utils_func.pickle_dump_list_to_s3(top_n_col_list, self.folder_path, self.file_name, self.bucket_name, self.aws_region)

        #column list for dataframe
        cols_for_model_df_list = self.id_col_list + top_n_col_list
        df_feature_eng_output = df_input[cols_for_model_df_list]
        df_model_input = df_feature_eng_output.copy()

        push_status = utils_func.push_df_to_s3(df_model_input, self.bucket_name, self.aws_region, self.file_path, self.s3_object_key)
        print(push_status)

if __name__ == '__main__':
    with open('./conf/tasks/feature_pipepline.yml', 'r') as config_file:
        configuration = yaml.safe_load(config_file)
    task = DataPrep(configuration)
    task.preprocess_data()
