import boto3
import urllib
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from io import BytesIO
import pickle
import os
import importlib.util
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score

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

    
    def load_pickle_from_s3(self, bucket_name, aws_region, file_path):
        try:
            # Create an S3 client
            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

            s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                      aws_secret_access_key=aws_secret_key, 
                      region_name=aws_region)
                

            s3_object = s3.Object(bucket_name, file_path)

            # Read the pickle file from the S3 response
            pickle_data = s3_object.get()['Body'].read()

            # Deserialize the pickle data to obtain the Python object (list in this case)
            loaded_list = pickle.loads(pickle_data)

            return loaded_list
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
        
            
    def convert_columns_to_string(self,df, columns):
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
            else:
                print(f"Column '{col}' not found in the DataFrame.")
                
    def eval_cm(self,model, X_train, y_train, X_val, y_val, drop_id_col_list):
        
        model.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)
        y_pred_train = model.predict(X_train.drop(drop_id_col_list, axis=1, errors='ignore'))
        y_pred_val = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_val = confusion_matrix(y_val, y_pred_val)
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix (Train)')
        plt.savefig('confusion_matrix_train.png')
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix (Validation)')
        plt.savefig('confusion_matrix_validation.png')
        
    def roc_curve(self,model, X_val,y_val, drop_id_col_list):
            
            """
            Logs Roc_auc curve in MLflow.

            Parameters:
            - y_test: The true labels (ground truth).
            

            Returns:
            - None
            """
            y_pred = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))
            fpr, tpr, thresholds = roc_curve(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_pred)

            # Create and save the ROC curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            roc_curve_plot_path = "roc_curve.png"
            
            plt.savefig(roc_curve_plot_path)
            
    def evaluation_metrics(self, model, X_train, y_train, X_val, y_val, drop_id_col_list):
        
        """
            Logs f1_Score and accuracy in MLflow.

            Parameters:
            - y_test: The true labels (ground truth).
            - y_pred: The predicted labels (model predictions).
            - run_name: The name for the MLflow run.

            Returns:
            - f1score and accuracy
        """

        model.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)
        y_pred_train = model.predict(X_train.drop(drop_id_col_list, axis=1, errors='ignore'))
        y_pred_val = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))

        f1_train = f1_score(y_train, y_pred_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        f1_val = f1_score(y_val, y_pred_val)
        accuracy_val = accuracy_score(y_val, y_pred_val)

        return {'Train_F1-score' : round(f1_train,2),
                'Validation_F1-score' : round(f1_val,2),
                'Train_Accuracy' : round(accuracy_train,2),
                'Validation_Accuracy' : round(accuracy_val,2)}
