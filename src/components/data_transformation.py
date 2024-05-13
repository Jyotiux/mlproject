import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)


# import sys
# import os
# from dataclasses import dataclass
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,StandardScaler

# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_object


# # By using the @dataclass without using the constructor __init__(), the class (DataTransformationConfig ) accepted the value and assigned to the given variable, so that in this case automatically the 'preprocessor.pkl' file will be created in the 'artifacts' folder... 
# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")


# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config=DataTransformationConfig()

#     def get_data_transformer_object(self):
#         # "this function is responsible for data transformation"
#         try:
#             numerical_columns = ["writing_score", "reading_score"]
#             categorical_columns = [
#                 "gender",
#                 "race_ethnicity",
#                 "parental_level_of_education",
#                 "lunch",
#                 "test_preparation_course",
#             ]

#             num_pipeline=Pipeline(
#                 steps=[
#                     "imputer",SimpleImputer(strategy="median"),("scalar",StandardScaler())
#                 ]
#             )
#             cat_pipeline=Pipeline(
#                 steps=[
#                     ("imputer",SimpleImputer(strategy="most_frequent")),
#                     ("onehotencoder",OneHotEncoder()),("scalar",StandardScaler())
#                 ]
#             )
#             logging.info(f"Numerical columns standard scaling completed")
#             logging.info(f"Categorical columns encoding completed")

#             preprocessor=ColumnTransformer(
#                 [
#                     ("num_pipeline",num_pipeline,numerical_columns),("cat_pipeline",cat_pipeline,categorical_columns)
#                 ]
#             )
#             return preprocessor
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def initiate_data_transformation(self, train_path,test_path):
#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)

#             logging.info(f"Read train and test data complete")
#             logging.info(f"Obtaining preprocessing object")
#             preprocessing_obj=self.get_data_transformer_object()

#             target_column_name="math_score"
#             numerical_columns=['writing_score','reading_score']

#             target_feature_train_df=train_df[target_column_name]
#             input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
           

#             target_feature_test_df=test_df[target_column_name]
#             input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            


#             logging.info(
#                 f"Applying preprocessing object on training dataframe and testing dataframe."
#             )

#             input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

#             train_arr = np.c_[
#                 input_feature_train_arr, np.array(target_feature_train_df)
#             ]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             logging.info(f"Saved preprocessing object.")

#             save_object(

#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj

#             )

#             return (
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path,
#             )
#         except Exception as e:
#             raise CustomException(e,sys)
            
# # Onehotencoding for converting categorical variables similar to dummy variables in pandas. 

# # Standard scalar for scaling the numerical variables is a similar scale, another scalar is the minmax scaler.























# import pandas as pd
# import os

# # Define the path to the artifacts folder
# artifacts_folder = "artifacts/"

# # Define the filenames for train and test data
# train_filename = "train.csv"
# test_filename = "test.csv"

# # Construct the full paths to the train and test files
# train_filepath = os.path.join(artifacts_folder, train_filename)
# test_filepath = os.path.join(artifacts_folder, test_filename)

# # Read train and test data into pandas DataFrames
# train_data = pd.read_csv(train_filepath)
# test_data = pd.read_csv(test_filepath)
# columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course",
#            "math_score", "reading_score", "writing_score"]
# # gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,math_score,reading_score,writing_score
# # Create DataFrame
# train_data = pd.DataFrame(train_data, columns=columns)


# # Now you can use train_data and test_data DataFrames for further processing
# # Check for NaN values in train data
# train_nan_values = train_data.isna().sum()

# # Check for NaN values in test data
# test_nan_values = test_data.isna().sum()

# print("NaN values in train data:")
# print(train_nan_values)

# print("\nNaN values in test data:")
# print(test_nan_values)

# # Assuming you have a DataFrame named train_data
# # train_data['math_score'].isna().any()
# # train_data['reading_score'].isna().any()
# # train_data['writing_score'].isna().any()


# # # Calculate 50% of maximum score for each subject
# # math_threshold = train_data['math_score'].max() * 0.5
# # reading_threshold = train_data['reading_score'].max() * 0.5
# # writing_threshold = train_data['writing_score'].max() * 0.5
# # print(math_threshold,reading_threshold,writing_threshold)

# # # Filter the original dataset to get only those rows where all scores are above 50%
# # filtered_data = train_data[
# #     (train_data['math_score'] > math_threshold) &
# #     (train_data['reading_score'] > reading_threshold) &
# #     (train_data['writing_score'] > writing_threshold)
# # ]

# # # Now filtered_data contains the subset of train_data where all scores are above 50%
# # print(filtered_data)