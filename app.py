import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


model=load_model('next_word_lstm.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer=pickle.load(handle)

def predict_next(model,tokenizer,text,max_seq_len):
    token_list=tokenizer.textsst_to_sequences([text])[0]
    if len(token_list)>=max_seq_len:
        token_list=token_list[-(max_seq_len-1):]
    token_list=pad_sequences([token_list],max_seq_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_next=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_next:
            return word
    return None
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

