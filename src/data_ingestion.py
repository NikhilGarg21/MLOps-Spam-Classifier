import pandas as pd
import os
from sklearn.model_selection import train_test_split
from logger import get_logger
import yaml

logger = get_logger("data_ingestion")
logger.info("Starting data ingestion")

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

def save_data(train_data : pd.DataFrame , test_data : pd.DataFrame , data_path : str) -> None:
    """Save the train , test data locally"""
    try:
        raw_data_path = os.path.join(data_path , 'raw')
        os.makedirs(raw_data_path , exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path , "train.csv") , index = False)
        test_data.to_csv(os.path.join(raw_data_path , "test.csv") , index = False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        test_size = 0.2
        data_path = r"C:\MLOPS\dummy\sms_spam.csv"
        df = load_data(data_url=data_path)
        train_data , test_data = train_test_split(df, test_size=test_size , random_state=42)
        save_data(train_data=train_data , test_data=test_data , data_path="./data")
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

