import pandas as pd
import numpy as np
from typing import Tuple
import sys
from pathlib import Path
from src.logger import CustomLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from yaml import safe_load
import logging


TARTGET = 'sentiment'

# custom logger for module
logger = CustomLogger('feature_engineering')

# create a stream handler
console_handler = logging.StreamHandler()

# add console handler to the logger
logger.logger.addHandler(console_handler)

# read the dataframe
def read_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df


# make X and y
def make_X_and_y(df: pd.DataFrame,target_column) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_column).squeeze(axis='columns')
    y = df[target_column]
    logger.log_message('Data split into X and y')
    return X, y


def fit_tfidf_transformer(X_train: pd.Series, max_features: int):
    tf_idf = TfidfVectorizer(max_features=max_features)
    # fit the transformer on training data
    tf_idf.fit(X_train)
    logger.log_message('Tfidf transformer trained')
    return tf_idf


def transform_data(X,transformer):
    X_trans = pd.DataFrame(transformer.transform(X).toarray())
    logger.log_message('Data tansformed through transformer')
    logger.log_message(f'Shape of data after transformation is {X_trans.shape}')
    return X_trans


def fit_label_encoder(y):
    encoder = LabelEncoder()
    # fit the encoder
    encoder.fit(y=y)
    logger.log_message('Encoder trained')
    return encoder


def encode_label(y,encoder):
    y_trans = encoder.transform(y)
    logger.log_message('Feature Encoded')
    return y_trans


def save_the_data(dataframe: pd.DataFrame, data_path: Path) -> None:
    try:
        # save the data to path
        dataframe.to_csv(data_path,index=False)
        logger.log_message(f"DataFrame {data_path.name} saved successfuly")
    except Exception as e:
        logger.log_message(f'Dataframe not saved')

        
def read_parameters(params_file_path: str) -> dict:
    with open(params_file_path,'r') as params:
        parameters = safe_load(params)
    logger.log_message('Parameters read successfully from params file')
    return parameters


def save_transformer(obj, save_path: Path):
    joblib.dump(value=obj,filename=save_path)


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna()
        )


def main():
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "interim"
    save_data_path = data_path.parent / "processed"
    save_data_path.mkdir(exist_ok=True)
    logger.log_message("Processed data folder created")
    save_transformer_path = root_path / "models" / "transformers"
    save_transformer_path.mkdir(exist_ok=True)
    logger.log_message("Transformers folder created")
    
    
    for ind in range(1,3):
        filename = sys.argv[ind]
        # read the df
        df = read_data(data_path / filename)
        logger.log_message(f'{filename} read successfuly')
        # remove missing values from data
        df = drop_missing_values(df)
        max_features = read_parameters('params.yaml')['feature_engineering']['max_features']
        logger.log_message(f'The max features value read from params file is {max_features}')
        
        X, y = make_X_and_y(df,TARTGET)
        
        if filename == 'train_processed.csv':
            # tfidf fit on data
            tf_idf = fit_tfidf_transformer(X,max_features)
            # transform the data
            X_trans = transform_data(X,tf_idf)
            # save transformer
            save_transformer(tf_idf,save_transformer_path / "tf_idf.joblib")
            # fit encoder on target
            encoder = fit_label_encoder(y)
            # transform the target
            y_trans = encode_label(y,encoder)
            # save the encoder 
            save_transformer(encoder, save_transformer_path / "label_encoder.joblib")
            # concatenate the data
            X_trans['target'] = y_trans
            logger.log_message(f'Shape of the final dataframe is {X_trans.shape}')
            # save the data
            save_the_data(X_trans,save_data_path / f"{filename.replace("_processed","_final")}")

        elif filename == "test_processed.csv":
            # transform the data
            X_trans = transform_data(X,tf_idf)
            # transform the target
            y_trans = encode_label(y,encoder)
            # concatenate the data
            X_trans['target'] = y_trans
            logger.log_message(f'Shape of the final dataframe is {X_trans.shape}')
            # save the data
            save_the_data(X_trans,save_data_path / f"{filename.replace("_processed","_final")}")



if __name__ == "__main__":
    main()
    
    