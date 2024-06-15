# Importing necessary libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

path = 'spamEmails.csv'

data = pd.read_csv(path)


def removePunctuation(Message):
    temp = str.maketrans('', '', punctuationList)
    return Message.translate(temp)


def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def removeStopWords(Message):

    ImportantWords = []

    for word in str(Message).split():
        word = word.lower()

        if word not in stopWords:
            ImportantWords.append(word)

    output = " ".join(ImportantWords)
    return output


def plotWordCloud(data, typ):
    emailCorpus = ''.join(data['Message'])

    plt.figure(figsize=(7, 7))

    wc = WordCloud(background_color='black',
                   max_words=100,
                   width=800,
                   height=400,
                   collocations=False).generate(emailCorpus)
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {typ} emails', fontsize=15)
    plt.axis('off')
    plt.show()


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
if len(HamMsg) > len(SpamMsg):
    HamMsg = HamMsg.sample(n=len(SpamMsg), random_state=42)  # Downsamples the ham messages to balance the samples
if len(SpamMsg) > len(HamMsg):
    SpamMsg = SpamMsg.sample(n=len(HamMsg), random_state=42)

BalancedData = pd.concat([HamMsg, SpamMsg])
plt.figure(figsize=(8, 6))
sns.countplot(data=BalancedData, x='Spam')
plt.title('Distribution of Ham and Spam Emails after Balancing Sample sizes')
plt.xlabel('Message Types')
#plt.show()  # Completely equal split between spam data and ham data
print(BalancedData)

punctuationList = string.punctuation  # Creates a str with all punctuation
BalancedData['Message'] = BalancedData['Message'].apply(lambda x: removePunctuation(x))

stopWords = stopwords.words('english')
BalancedData['Message'] = BalancedData['Message'].apply(lambda x: removeStopWords(x))

BalancedData['Message'] = BalancedData['Message'].apply(lambda x: normalize_text(x))
print(BalancedData.head())  # prints first few entries

plotWordCloud(BalancedData[BalancedData['Spam'] == 0], typ='Non-Spam')
plotWordCloud(BalancedData[BalancedData['Spam'] == 1], typ='Spam')

train_X, test_X, train_Y, test_Y = train_test_split(BalancedData['Message'],
                                                    BalancedData['Spam'],
                                                    test_size=0.2,
                                                    random_state=42)

tokenizer = Tokenizer()  # init the tokenizer
tokenizer.fit_on_texts(train_X)  # fits the tokeniser to the training X.

train_sequences = tokenizer.texts_to_sequences(train_X)  # creates a sequence of tokens for the training data
test_sequences = tokenizer.texts_to_sequences(test_X)  # does same for test

max_len = 100  # maximum sequence length
train_sequences = pad_sequences(train_sequences,
                                maxlen=max_len,
                                padding='post',  # zeros are added as padding
                                truncating='post')  # sequences longer than max_length are truncated
test_sequences = pad_sequences(test_sequences,
                               maxlen=max_len,
                               padding='post',
                               truncating='post')

train_sequences = np.array(train_sequences, dtype=np.int32)
test_sequences = np.array(test_sequences, dtype=np.int32)
train_Y = np.array(train_Y, dtype=np.int32)
test_Y = np.array(test_Y, dtype=np.int32)


model = tf.keras.models.Sequential()  # Builds the model
# Convert input sequences of word indices into dense vectors of fixed size.
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                    output_dim=32,
                                    input_length=max_len))
model.add(tf.keras.layers.LSTM(16))  # capture temporal dependencies in the input sequences. 16 dimension output
model.add(tf.keras.layers.Dense(32, activation='relu'))  # Fully connected layer for processing after the LSTM
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer to produce the final classification.


model.compile(loss=tf.keras.losses.BinaryCrossentropy(),  # passed through activation func before output
              metrics=['accuracy'],  # Proportion of correct classification to incorrect
              optimizer='adam')  # Adaptive Moment Estimation
model.build(input_shape=(None, max_len))
model.summary()

es = EarlyStopping(patience=2,  # Stops early if there is no improvement in learning after 3 epochs
                   monitor='val_loss',  # validation accuracy is monitored
                   restore_best_weights=True)

lr = ReduceLROnPlateau(patience=3,
                       monitor='val_loss',
                       factor=0.5,
                       verbose=0)  # Message not printed when LR is reduced
history = model.fit(train_sequences, train_Y,
                    validation_data=(test_sequences, test_Y),
                    epochs=20,
                    batch_size=32,
                    callbacks=[lr, es])
# Train the model


# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('Test Loss :', test_loss)
print('Test Accuracy :', test_accuracy)


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()