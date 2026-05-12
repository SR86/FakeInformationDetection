import pandas as pd 
import matplotlib.pyplot as plt
import re
import string
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data():
    # Load true news data
    true_df = pd.read_csv('data/ISOT dataset/True.csv')[['title', 'text']]

    # Load fake news data
    fake_df = pd.read_csv('data/ISOT dataset/Fake.csv')[['title', 'text']]

    # Label the data
    true_df['target'] = 0  # True news as label 0
    fake_df['target'] = 1  # Fake news as label 1

    # Combine both datasets
    df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
    df = df.drop_duplicates(subset=['title', 'text'])  # Remove duplicates

    return df



def plot_distribution(df):
    df.target.value_counts(normalize=True).plot(kind='bar')
    plt.title('Target Distribution')
    plt.xlabel('Target')
    plt.ylabel('Frequency')
    plt.show()

def text_preprocessing(text):
    # Check if the input is a string; if not, convert it to an empty string
    if not isinstance(text, str):
        text = ''
        
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation + "–—−±×÷"), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    text = re.sub(r'reuters', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text

def preprocess_data(df):
    df['title'] = df['title'].apply(text_preprocessing)
    df['text'] = df['text'].apply(text_preprocessing)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def split_and_save(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    train_df.to_csv('./data/ISOT dataset/train.csv', index=False)
    test_df.to_csv('./data/ISOT dataset/test.csv', index=False)
    print("Train dataset size : ", len(train_df))
    print("Test dataset size : ", len(test_df))

if __name__ == "__main__":
    df = load_data()
    print("News dataset size : ", len(df), end= "\n")
    plot_distribution(df)
    df = preprocess_data(df)
    print(df.head(), end = "\n\n")
    split_and_save(df)
    print("News data preprocessing is done.")
