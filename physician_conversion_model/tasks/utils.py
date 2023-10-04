import boto3
import urllib
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from io import BytesIO
import pickle
import os
import importlib.util
import sys

class utils():
   

    def push_df_to_s3(self, df, bucket_name, aws_region, s3_object_key):
        """
        Push a DataFrame to an S3 bucket.

        Args:
            df (pd.DataFrame): The DataFrame to push to S3.
            bucket_name (str): The name of the S3 bucket.
            aws_region (str): The AWS region of the S3 bucket.
            file_path (str): The path to the S3 file.
            s3_object_key (str): The key of the S3 object.

        Returns:
            dict: A dictionary with a status message.
        """
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        s3 = boto3.resource("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)

        s3.Object(bucket_name, s3_object_key).put(Body=csv_content)

        return {"df_push_status": 'success'}
    

    def load_data_from_s3(self, bucket_name, aws_region, file_path):
        """
        Load data from an S3 bucket.

        Args:
            bucket_name (str): The name of the S3 bucket.
            aws_region (str): The AWS region of the S3 bucket.
            file_path (str): The path to the S3 file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        s3 = boto3.resource("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)

        s3_object = s3.Object(bucket_name, file_path)
        csv_content = s3_object.get()['Body'].read()

        df_input = pd.read_csv(BytesIO(csv_content))

        return df_input
    

    def select_kbest_features(self, df, target_col, n):
        """
        Select the top n features from a DataFrame using the SelectKBest algorithm.

        Args:
            df (pd.DataFrame): The DataFrame to select features from.
            target_col (pd.Series): The target column for feature selection.
            n (int): The number of features to select.

        Returns:
            pd.Index: A list of the top n features.
        """
        selector = SelectKBest(score_func=f_classif, k=n)
        selected_features = selector.fit(df, target_col)
        mask = selector.get_support()
        top_n_features = df.columns[mask]

        return top_n_features

        
        
    def pickle_dump_list_to_s3(self, column_list, folder_path, file_name, bucket_name, aws_region):
        """
        Pickle dump a list of columns and upload it to an S3 bucket.

        Args:
            column_list (list): List of columns to pickle.
            folder_path (str): The path within the S3 bucket where the file will be stored.
            file_name (str): The name of the pickled file.
            bucket_name (str): The name of the S3 bucket.
            aws_region (str): The AWS region of the S3 bucket.
        """
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        # Pickle dump the list
        with open(file_name, 'wb') as file:
            pickle.dump(column_list, file)

        # Upload the pickled file to S3
        s3 = boto3.resource("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
        s3.Bucket(bucket_name).upload_file(file_name, os.path.join(folder_path, file_name))

        

