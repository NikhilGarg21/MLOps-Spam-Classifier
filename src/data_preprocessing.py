import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
from logger import get_logger
import os 

logger = get_logger("data_preprocessing")
logger.info("Starting data preprocessing")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preprocess(df , text_column='text', target_column='label'):
    """
    Preprocesses the DataFrame by transforming the text column and converting target column into numbers(spam - 1 / not spam - 0).
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        df[target_column] = df[target_column].map({'ham': 0, 'spam': 1})
        logger.debug('Target column encoded')
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def main(text_column = "text" , target_column = "label"):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        train_data = pd.read_csv(r"C:\MLOPS\data\raw\train.csv")
        test_data = pd.read_csv(r"C:\MLOPS\data\raw\test.csv")
        logger.debug("Data loaded successfully")

        train_preprocess_data = preprocess(df = train_data , text_column=text_column , target_column=target_column)
        test_preprocess_data = preprocess(df = test_data , text_column=text_column , target_column=target_column)

        data_path = os.path.join("./data" , "interim")
        os.makedirs(data_path , exist_ok=True)

        train_preprocess_data.to_csv(os.path.join(data_path , "train_preprocess.csv") , index = False)
        test_preprocess_data.to_csv(os.path.join(data_path , "test_preprocess.csv") , index = False)

        logger.debug("file saved successfully")

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()