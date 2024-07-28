import pandas as pd
import logging
import sys
from src.logger import CustomLogger
import joblib
from pathlib import Path
from typing import Tuple
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


TARGET = 'target'
MODEL_NAME = 'rf.joblib'

# custom logger for module
logger = CustomLogger('predict')

# create a stream handler
console_handler = logging.StreamHandler()

# add console handler to the logger
logger.logger.addHandler(console_handler)

# read the dataframe
def read_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.log_message(f'{data_path.name} does not exist')
    else:
        return df


# make X and y
def make_X_and_y(df: pd.DataFrame,target_column) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_column)
    y = df[target_column]
    logger.log_message('Data split into X and y')
    return X, y

        
def do_predictions(model,X):
    y_pred = model.predict(X)
    return y_pred


def load_model(model_path: Path):
    try:
        clf = joblib.load(model_path)
    except FileNotFoundError as e:
        logger.log_message("Model Path does not exist")
    else:
        logger.log_message("Model Loaded from path successfully")
        return clf
    

        
def calculate_metrics(data_name: str,metrics_obj: dict,y,y_pred):
    accuracy = accuracy_score(y,y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(y,y_pred,
                                                                           average=None,
                                                                           zero_division=0.0)
    
    metrics_obj[data_name] = {'accuracy':accuracy,
                              'precision':precision.tolist(),
                              'recall':recall.tolist(),
                              'f1_score':f1_score.tolist(),
                              'support':support.tolist()}
    
    return metrics_obj


def main():
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "processed"    
    model_load_path = root_path / "models" / "classifiers"
    metrics_save_path = root_path / "reports" / "metrics.json"
    metrics_dict = {}
    
    for ind in range(1,3):
        filename = sys.argv[ind]
        # read the data
        df = read_data(data_path=data_path / filename)
        # split into x and y
        X,y = make_X_and_y(df=df,target_column=TARGET)
        # load the model
        clf = load_model(model_path=model_load_path / MODEL_NAME)
        # get predictions
        y_pred = do_predictions(model=clf,X=X)
        # generate metrics
        metrics_dict = calculate_metrics(data_name=(data_path / filename).name,
                                         metrics_obj=metrics_dict,
                                         y=y,y_pred=y_pred)
        
    # convert metrics dict to json
    with open(metrics_save_path,"w") as f:
        json.dump(metrics_dict,f,indent=4)
    
    logger.log_message("Metrics saved to save location")


if __name__ == "__main__":
    main()


