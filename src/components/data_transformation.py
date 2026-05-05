import sys
import os
import re
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_data_path: str=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_trasnformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Responsible for mapping specific transformation to dataset columns.
        '''
        try:
            text_columns = ["Resume", "Job Description"]
            categorical_columns = ["Gender", "Race", "Ethnicity", "Job Roles"]
            numerical_columns = ["Age"]

            text_pipeline = Pipeline(
                steps = [
                    ("tfidf", TfidfVectorizer(max_features=500, stop_words='english'))
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            num_pipeline = Pipeline(
                steps = [
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers = [
                    ("text_resume", text_pipeline, text_columns[0]),
                    ("text_jd", text_pipeline, text_columns[1]),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def balance_data(self, df):
        '''
        Balances the Gender Bias: Downsamples successful Male matches to match the count of successful Female matches.
        '''

        try:
            logging.info('Starting dataset balancing to mitigate gender bias.')

            male_success = df[(df['Gender'] == 'Male') & (df['Best Match'] == 1)]
            female_success = df[(df['Gender'] == 'Female') & (df['Best Match'] == 1)]
            others = df[~((df['Gender'] == 'Male') & (df['Best Match'] == 1))]

            male_success_downsampled = resample(
                male_success,
                replace=False,
                n_samples=(len(female_success)),
                random_state=42
            )

            balanced_df = pd.concat([others, male_success_downsampled])

            logging.info(f"Balanced counts - Male Matches: {len(male_success_downsampled)}")
            return balanced_df
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            train_df = self.balance_data(train_df)

            target_column_name = "Best Match"
            drop_columns = [target_column_name, "Job Applicant Name"]

            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info('Preprocessing object has been obtained.')

            save_object(
                file_path=self.data_trasnformation_config.preprocessor_obj_data_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_trasnformation_config.preprocessor_obj_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

