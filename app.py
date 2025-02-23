import pandas as pd
import numpy as np
from word2number import w2n
import re
from dateutil import parser
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


###### Download necessary NLTK data #####
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


##### TO-DO #####
service_account_key = "path/to/service_account_key.json"
bucket-name = "path/to/bucket"


###### Authentication ######
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key


###### Extracting data from source #######
df_og = pd.read_json('sample_data.json')
df = df_og.copy()


###### Raw data transformation functions ######

# Price cleaning
def price_transform(value):
    try:
        if isinstance(value, (int, float)):
            return round(float(value), 2)
        elif isinstance(value, str):
            try:
                return round(float(w2n.word_to_num(value)), 2)
            except:
                digits = re.findall(r'\d+', value)
                return round(float(''.join(digits)), 2) if digits else None
    except:
        return None

# Preprocessing textual columns
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Split hyphenated words
    text = text.replace('-', ' ')
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a single string
    return ' '.join(tokens)

# Function to add underscore prefix to column names starting with a number to comply with SQL-compatible column naming rules
def add_prefix_to_numeric_columns(column_name):
    if column_name[0].isdigit():
        return '_' + column_name
    return column_name


####### Transformation Layer #######

# Filling missing Quantity values
df['Quantity'] = df['Quantity'].replace('', None)
df['Quantity'] = df['Quantity'].astype(float)
df['Quantity'] = df['Quantity'].fillna(0)
df['Quantity'] = df['Quantity'].astype(int)

# Standardising Price values & filling nulls
df['Price'] = df['Price'].apply(price_transform)
df['Price'] = df['Price'].fillna(0)

# Standardising DateAdded format
df['DateAdded'] = df['DateAdded'].apply(lambda x: parser.parse(x, dayfirst=True))
df['DateAdded'] = pd.to_datetime(df['DateAdded'])
df['YearAdded'] = df['DateAdded'].dt.year.astype(int)
df['MonthAdded'] = df['DateAdded'].dt.month.astype(int)
df['DayAdded'] = df['DateAdded'].dt.day.astype(int)
df['Product_Age'] = df['DateAdded'].apply(lambda x: (pd.to_datetime('today') - x).days)
df["event_timestamp"] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')

# Exploding array lists
# df = df.explode('KeyFeatures').reset_index(drop=True)

# Ensuring all entries in KeyFeatures and Description columns are strings
df['KeyFeatures'] = df['KeyFeatures'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Dropping the DateAdded column
df.drop(columns=['DateAdded','data'], inplace=True)


#### Feature Engineering ####

df_main = df.copy()

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Initializing stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Label encoding the Brand & Category columns
df_main['Brand_le'] = label_encoder.fit_transform(df_main['Brand'])
df_main['Category_le'] = label_encoder.fit_transform(df_main['Category'])

# Filling missing values in KeyFeatures and Description
df_main['KeyFeatures'] = df_main['KeyFeatures'].fillna('')
df_main['Description'] = df_main['Description'].fillna('')

# Preprocessing to KeyFeatures and Description
df_main['KeyFeatures'] = df_main['KeyFeatures'].apply(preprocess_text)
df_main['Description'] = df_main['Description'].apply(preprocess_text)

# Initializing TF-IDF Vectorizer
tfidf_vectorizer_key = TfidfVectorizer(max_features=100)

# Vectorizing KeyFeatures
keyfeatures_tfidf = tfidf_vectorizer_key.fit_transform(df_main['KeyFeatures'])

# Getting actual feature names
keyfeature_names = tfidf_vectorizer_key.get_feature_names_out()

# # Add 'k' as prefix to the keyfeature_names
# keyfeature_names = [f'k_{name}' for name in keyfeature_names]

# Creating a DataFrame with the actual feature names as columns
keyfeatures_df = pd.DataFrame(keyfeatures_tfidf.toarray(), columns=keyfeature_names)

# Initializing TF-IDF Vectorizer
tfidf_vectorizer_desc = TfidfVectorizer(max_features=500)

# Vectorizing Description
description_tfidf = tfidf_vectorizer_desc.fit_transform(df_main['Description'])

# Getting actual feature names
description_names = tfidf_vectorizer_desc.get_feature_names_out()

# Creating a DataFrame with the actual feature names as columns
description_df = pd.DataFrame(description_tfidf.toarray(), columns=description_names)

# Identifying and dropping duplicate columns from the Description DataFrame
duplicate_columns = set(keyfeature_names).intersection(set(description_names))
description_df.drop(columns=duplicate_columns, inplace=True)

# Concatenate the TF-IDF features with the original DataFrame
df_main = pd.concat([df_main, keyfeatures_df, description_df], axis=1)

# Drop the original KeyFeatures and Description columns
df_main.drop(columns=['KeyFeatures', 'Description'], inplace=True)

# Apply SQL-compatible rule function to column names
df_main.columns = [add_prefix_to_numeric_columns(col) for col in df_main.columns]

# Creating a timestamp for feature creation to track lineage
df_main['event_timestamp'] = pd.to_datetime(df_main['event_timestamp'])


##### Uploading to Data Lake ##### 
df_main.to_csv(f'gs://{bucket-name}/feature_store.csv', index=False)