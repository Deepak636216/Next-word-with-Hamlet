
# Next Word Prediction with LSTM and Early Stopping

This project is a **Next Word Prediction** application built using an **LSTM** (Long Short-Term Memory) model in **TensorFlow/Keras** and deployed with **Streamlit**. The model is trained on **Shakespeare's Hamlet** text corpus and can predict the next word in a given sequence of words.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

This project demonstrates a text generation model using an LSTM neural network that predicts the next word in a sentence given a sequence of words. The model is trained on **Hamlet's** text file and deployed using a Streamlit web interface.

## Installation

### Prerequisites

- Python 3.x
- TensorFlow
- Streamlit
- Numpy
- Scikit-learn
- Pickle

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/next-word-prediction.git
   cd next-word-prediction


2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download or prepare the dataset** (e.g., `hamlet.txt`):
   - Add the `hamlet.txt` file to the project directory.

4. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

## Data Preparation

1. **Tokenization**:
   - We use `Tokenizer` from Keras to convert text into sequences of integers.

2. **Generate Input Sequences**:
   - The text is tokenized into sequences, and each line is split into n-grams.

3. **Pad Sequences**:
   - Input sequences are padded to ensure all sequences have the same length.

## Model Training

1. **Model Architecture**:
   - Embedding layer
   - Two LSTM layers with dropout for regularization
   - Dense layer with softmax activation for multi-class classification

2. **Early Stopping**:
   - Implemented to prevent overfitting by monitoring validation loss.

3. **Training**:
   - The model is trained with `categorical_crossentropy` loss and `adam` optimizer.
   - Training includes a validation split and the model's best weights are saved.

4. **Save Model and Tokenizer**:
   - The model and tokenizer are saved for later inference in the Streamlit app.

## Web Application

The Streamlit app provides a simple interface to input a sequence of words and get a predicted next word.

### Main Files:

- **app.py**: Streamlit application code
- **next_word_lstm.h5**: Saved LSTM model
- **tokenizer.pickle**: Saved tokenizer for preprocessing text

## Usage

1. Run the app:
   ```bash
   streamlit run app.py
   ```

2. Enter a sequence of words in the text input field and click **Predict Next Word** to get the next word prediction.

## Technologies Used

- **Python**: Programming language
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **Numpy**: Array manipulation library
- **Scikit-learn**: Data splitting
- **Pickle**: Serialization of tokenizer

## Project Structure

```plaintext
.
├── app.py                   # Streamlit app code
├── hamlet.txt               # Text corpus used for training
├── model_training.py        # Script for training the LSTM model
├── next_word_lstm.h5        # Saved LSTM model
├── tokenizer.pickle         # Saved tokenizer
└── README.md                # Project documentation
```
## URL
- https://next-word-with-hamlet-cbeerybcm5acfcvkbjdaru.streamlit.app/
```

