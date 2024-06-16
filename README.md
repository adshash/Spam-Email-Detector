# Email Spam Classification using LSTM

This repository contains code for building and training a Long Short-Term Memory (LSTM) neural network model to classify emails as either spam or non-spam (ham). The dataset used for training and testing the model is stored in a CSV file named `spamEmails.csv`.

## Project Structure

The project is structured as follows:

- `spamEmails.csv`: CSV file containing the email data with columns for `Category` (spam/ham) and `Message`.
- `README.md`: This document explaining the project and its components.
- `main.py`: Python script containing the code for data preprocessing, model building, training, and evaluation.

## Requirements

Ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `nltk`
- `wordcloud`
- `tensorflow`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn nltk wordcloud tensorflow scikit-learn
```

## Overview

The project performs the following steps:

1. **Data Loading and Preprocessing**:
   - Reads the CSV file (`spamEmails.csv`) containing email data.
   - Converts the `Category` column to binary labels (`spam` to 1, `ham` to 0).
   - Balances the dataset by downsampling the majority class (ham) to match the number of samples in the minority class (spam).
   - Removes punctuation, stopwords, and normalizes text using Unicode transformations.

2. **Exploratory Data Analysis (EDA)**:
   - Generates a word cloud visualization for both spam and non-spam emails to understand the most frequent words.

3. **Model Building**:
   - Tokenizes and sequences the text data.
   - Builds an LSTM-based neural network model using TensorFlow/Keras.
   - Compiles the model with binary cross-entropy loss, accuracy metric, and Adam optimizer.

4. **Model Training**:
   - Trains the LSTM model on the preprocessed data.
   - Uses early stopping and learning rate reduction on plateau as callbacks to improve training efficiency.

5. **Model Evaluation**:
   - Evaluates the trained model on the test set to measure its performance in terms of loss and accuracy.
   - Plots training and validation accuracy across epochs to visualize model performance.

## Usage

To run the code:

1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the required libraries as mentioned in the Requirements section.

3. Execute the `main.py` script:

   ```bash
   python main.py
   ```

4. The script will load the data, preprocess it, build the LSTM model, train it, evaluate its performance, and display relevant visualizations.

## Results

Upon running the script, you should see:

- The model summary detailing the architecture and parameters.
- Training logs showing progress and performance metrics.
- A plot displaying training and validation accuracy over epochs.
- Evaluation results including test loss and accuracy.

## Conclusion

This project demonstrates the application of LSTM neural networks for email spam classification. By preprocessing text data, balancing the dataset, and training an LSTM model, we achieve effective classification results on the provided dataset.
