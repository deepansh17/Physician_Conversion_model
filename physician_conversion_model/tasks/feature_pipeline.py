#system dependencies
import pandas as pd
import numpy as np


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import warnings
import os
import boto3
import urllib
import pickle

from io import BytesIO



#pyspark and feature store 
import os
import datetime

#warnings
warnings.filterwarnings('ignore')
import yaml 

class DataPrep(): 
    
    def __init__(self, conf):
        self.conf = conf
        
         
    def push_df_to_s3(self,df):

        # AWS credentials and region
        aws_region = self.conf['s3']['aws_region']
        bucket_name = self.conf['s3']['bucket_name']
        file_path = self.conf['s3']['file_path']

        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")


        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                    aws_secret_access_key=aws_secret_key, 
                    region_name=aws_region)

        s3_object_key = self.conf['preprocessed']['preprocessed_df_path'] 
        s3.Object(self.conf['s3']['bucket_name'], s3_object_key).put(Body=csv_content)

        return {"df_push_status": 'success'}
    

    def load_data_from_s3(self):

        # AWS credentials and region
        aws_region = self.conf['s3']['aws_region']
        bucket_name = self.conf['s3']['bucket_name']
        file_path = self.conf['s3']['file_path']

        

        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        
        access_key = aws_access_key 
        secret_key = aws_secret_key

        print(f"Access key and secret key are {access_key} and {secret_key}")

        
        
        encoded_secret_key = urllib.parse.quote(secret_key,safe="")

        s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                      aws_secret_access_key=aws_secret_key, 
                      region_name=aws_region)
                

        s3_object = s3.Object(bucket_name, file_path)
        
        csv_content = s3_object.get()['Body'].read()

        df_input = pd.read_csv(BytesIO(csv_content))

        return df_input
    

    def select_kbest_features(self, df, target_col,n):
        """
        Selects the top n features from the DataFrame using the SelectKBest algorithm.

        Args:
            df: The DataFrame to select features from.
            n: The number of features to select.

        Returns:
            A list of the top n features.
        """


        selector = SelectKBest(k=n)
        selected_features = selector.fit_transform(df, target_col)
        
        mask = selector.get_support()
        top_n_features = df.columns[mask]

        return top_n_features
        
        
    def pickle_dump_list_to_s3(self, column_list):
        """
        Pickle dump a list of columns and upload it to an S3 bucket in the specified folder.

        Args:
        - column_list: List of columns to pickle.

        Returns:
        - upload pickle list to s3
        """
        # AWS details

        bucket_name = self.conf['s3']['bucket_name']
        aws_region = self.conf['s3']['aws_region']
        folder_path = self.conf['preprocessed']['model_variable_list_file_path']
        file_name = self.conf['preprocessed']['model_variable_list_file_name']

        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        access_key = aws_access_key 
        secret_key = aws_secret_key
        print(f"Access key and secret key are {access_key} and {secret_key}")

        # Create an S3 client
        s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                      aws_secret_access_key=aws_secret_key, 
                      region_name=aws_region)

        # Pickle dump the list
        with open(file_name, 'wb') as file:
            pickle.dump(column_list, file)

        # Upload the pickled file to S3
        s3.Bucket(bucket_name).upload_file(file_name, folder_path + file_name)

        print(f"Pickled file '{file_name}' uploaded to S3 bucket '{bucket_name}' in folder '{folder_path}'.")
  
  
    def preprocess_data(self):
                
        df_input = self.load_data_from_s3()

        df_input = df_input.reset_index()

        #Clean column names
        df_input.columns = df_input.columns.str.strip()
        df_input.columns = df_input.columns.str.replace(' ', '_')

        #Drop unwanted column: "HCO Affiliation" - "Affiliation Type" is more valid column for us
        drop_col_list = self.conf['feature_transformation']['drop_column_list']
        df_input.drop(drop_col_list, axis= 1, inplace= True)

        #One hot encode categorical features
        encode_col_list = self.conf['feature_transformation']['one_hot_encode_feature_list']
        df_input = pd.get_dummies(df_input, columns=encode_col_list, drop_first=True)

        #Select variables for feature selection
        id_target_col_list = self.conf['feature_transformation']['id_target_col_list']
        col_for_feature_selection = df_input.columns.difference(id_target_col_list)

        #Variance threshold feature selection method
        threshold = self.conf['param_values']['variance_threshold_value']
        var_thr = VarianceThreshold(threshold = threshold) #Removing both constant and quasi-constant
        var_thr.fit(df_input[col_for_feature_selection])
        #var_thr.get_support()

        df_input_subset = df_input[col_for_feature_selection]
        remove_col_list = [col for col in df_input_subset.columns 
                        if col not in df_input_subset.columns[var_thr.get_support()]]
        
        #remove above list column from master dataframe
        df_input.drop(remove_col_list, axis = 1, inplace = True, errors= 'ignore')

        #Feature Selection Using Select K Best
        n = self.conf['param_values']['select_k_best_feature_num']
        id_col_list = self.conf['feature_transformation']['id_col_list']
        target_col = self.conf['feature_transformation']['target_col']
        
        df = df_input.drop(id_col_list,axis=1)
        target_col_var = df_input[target_col]
        top_n_col_list = self.select_kbest_features(
                df,target_col_var, n)
        
        #Convert to list
        top_n_col_list = top_n_col_list.tolist()

        # Dump top_n_col_list to s3 bucket
        self.pickle_dump_list_to_s3(top_n_col_list)
        
        #column list for dataframe
        cols_for_model_df_list = id_col_list + top_n_col_list
        df_feature_eng_output = df_input[cols_for_model_df_list]
        df_model_input = df_feature_eng_output.copy()
        
        push_status = self.push_df_to_s3(df_model_input)
        print(push_status)





if __name__ == '__main__':
    with open('conf/tasks/feature_pipeline.yml', 'r') as config_file:
        configuration = yaml.safe_load(config_file)
    data_prep = DataPrep(configuration)
    data_prep.preprocess_data()
