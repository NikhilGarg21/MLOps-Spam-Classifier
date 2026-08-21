import os
from logger import get_logger
import pandas as pd
import pickle
from sklearn.svm import LinearSVC
import numpy as np 
import yaml


logger = get_logger("model_building")
logger.debug("Model_building started")

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        params = params["model_building"]
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url : str) -> pd.DataFrame:
    """load data from a csv file"""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s' , data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train : np.ndarray , y_train : np.ndarray , params : dict) -> LinearSVC:
    """
    Train the LinearSVC model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """

    try:
        if X_train.shape[0] != y_train.shape[0] :
            raise ValueError("The number of samples in X_train and y_train must be the same.")

        logger.debug("initialising Svm with parameters : %s" , params)
        model = LinearSVC(**params)

        logger.debug('Model training started with %d samples', X_train.shape[0])
        model.fit(X_train , y_train)
        logger.debug('Model training completed')
        
        return model
     
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model  , file_path : str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """

    try:
        os.makedirs(os.path.dirname(file_path) , exist_ok=True)

        with open (file_path , "wb") as file:
            pickle.dump(model , file)
            logger.debug('Model saved to %s', file_path)

    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def main():
    try:
        params = load_params(params_path = "params.yaml")
        train_data = load_data(r"C:\MLOPS\data\processed\train_tfidf.csv")
        X_train = train_data.iloc[ : , :- 1].values
        y_train = train_data.iloc[ :, -1].values
        model = train_model(X_train=X_train , y_train=y_train , params=params)
        model_save_path = "models/model.pkl"
        save_model(model , model_save_path)
        
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
    

     