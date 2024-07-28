import pandas as pd
import numpy as np
import re
import nltk
import sys
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import CustomLogger


nltk.download('wordnet')
nltk.download('stopwords')

# custom logger for module
logger = CustomLogger('text_preprocessing')

# create a stream handler
console_handler = logging.StreamHandler()

# add console handler to the logger
logger.logger.addHandler(console_handler)


# read the dataframe
def read_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df

nltk.download('wordnet')
nltk.download('stopwords')


def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)


def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text


def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)


def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()


def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(df):
    df.loc[:,'content'] = df.loc[:,'content'].apply(lambda x : lower_case(x))
    logger.log_message("All text converted to lowercase")
    df.loc[:,'content'] = df.loc[:,'content'].apply(lambda x : remove_stop_words(x))
    logger.log_message("Stop words removed")
    df.loc[:,'content'] = df.loc[:,'content'].apply(lambda x : removing_numbers(x))
    logger.log_message("Digits removed from text")
    df.loc[:,'content'] = df.loc[:,'content'].apply(lambda x : removing_punctuations(x))
    logger.log_message("All punctuations removed")
    df.loc[:,'content'] = df.loc[:,'content'].apply(lambda x : removing_urls(x))
    logger.log_message("Url removed from text")
    df.loc[:,'content'] = df.loc[:,'content'].apply(lambda x : lemmatization(x))
    logger.log_message("Lemmatization applied on text")
    return df


def save_the_data(dataframe: pd.DataFrame, data_path: Path) -> None:
    try:
        # save the data to path
        dataframe.to_csv(data_path,index=False)
        logger.log_message(f"DataFrame {data_path.name} saved successfuly")
    except Exception as e:
        logger.log_message(f'Dataframe not saved')
        

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna()
        )



def main():
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "raw"
    save_path = data_path.parent / "interim"
    save_path.mkdir(exist_ok=True)
    logger.log_message("Processed data folder created")
   
    for i in range(1,3):
        filename = sys.argv[i]
        # read the df
        df = read_data(data_path / filename)
        logger.log_message(f'{filename} read successfuly')
        # do preprocessing
        df_trans = normalize_text(df)
        logger.log_message(f'Text preprocessing on {filename} complete')
        # drop missing values
        df_final = drop_missing_values(df_trans) 
        logger.log_message(f'Missing values in the data are {df_final.isna().sum().sum()}')
        # save filename
        save_filename = filename.rstrip(".csv")
        # save the processed data
        save_the_data(df_final, save_path / f'{save_filename}_processed.csv')



if __name__ == "__main__":
    main()
