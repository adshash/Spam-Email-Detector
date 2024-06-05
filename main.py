# Importing necessary libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

path = 'spamEmails.csv'

data = pd.read_csv(path)

def removePunctuation(Message):
    temp = str.maketrans('', '', punctuationList)
    return Message.translate(temp)


def removeStopWords(Message):

    ImportantWords = []

    for word in str(Message).split():
        word = word.lower()

        if word not in stopWords:
            ImportantWords.append(word)

    output = " ".join(ImportantWords)
    return output

for index, row in data.iterrows():  # Changes the spam category to bool (1 or 0)
    if row['Category'] == 'spam':
        row['Category'] = 1
    elif row['Category'] == 'ham':
        row['Category'] = 0

data.rename(inplace=True, columns={'Category': 'Spam'})

print(data.shape)
sns.countplot(x='Spam', data=data)
plt.show()  # Shows how much spam vs ham emails there are

HamMsg = data[data.Spam == 0]
SpamMsg = data[data.Spam == 1]
HamMsg = HamMsg.sample(n=len(SpamMsg), random_state=37)  # Downsamples the ham messages to balance the samples

BalancedData = HamMsg._append(SpamMsg)\
    .reset_index(drop=True)
plt.figure(figsize=(8, 6))
sns.countplot(data=BalancedData, x='Spam')
plt.title('Distribution of Ham and Spam Emails after Balancing Sample sizes')
plt.xlabel('Message Types')
plt.show()  # Completely equal split between spam data and ham data
print(BalancedData)

punctuationList = string.punctuation  # Creates a str with all punctuation
BalancedData['Message'] = BalancedData['Message'].apply(lambda x: removePunctuation(x))

stopWords = stopwords.words('english')

BalancedData['Message'] = BalancedData['Message'].apply(lambda x: removeStopWords(x))
print(BalancedData.head())  # prints first few entries
