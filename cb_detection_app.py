import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit import components

# Data Manipulation
import numpy as np
import pandas as pd
import os
from PIL import Image

# Preprocessing Pipeline
import pandas as pd
import re
import unicodedata
import string
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import language_tool_python
import pickle  
import json
import re
import unicodedata
import string
import nltk
import spacy
import pickle
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from lime.lime_text import LimeTextExplainer
import preprocess_text as pt
import torch

# Configuration and Model Loading
pd.set_option('display.max_columns', None)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
tool = language_tool_python.LanguageTool('en-US')

#  Bad word mapping function
def create_profane_mapping(profane_words,vocabulary):
        '''
        Function creates a mapping between commonly found profane words and words in 
        document corpus 
        '''
        
        # mapping dictionary
        mapping_dict = dict()
        
        # looping through each profane word
        for profane in tqdm(profane_words):
            mapped_words = set()
            
            # looping through each word in vocab
            for word in vocabulary:
                # mapping only if ratio > 80
                try:
                    if fuzz.ratio(profane,word) > 90:
                        mapped_words.add(word)
                except:
                    pass
                    
            # list of all vocab words for given profane word
            mapping_dict[profane] = mapped_words
        
        return mapping_dict
    
def replace_words(corpus,mapping_dict):
        '''
        Function replaces obfuscated profane words using a mapping dictionary
        '''
        
        processed_corpus = []
        
        # iterating over each document in the corpus
        for document in tqdm(corpus):
            
            # splitting sentence to word
            comment = document.split()
            
            # iterating over mapping_dict
            for mapped_word,v in mapping_dict.items():
                
                # comparing target word to each comment word 
                for target_word in v:
                    
                    # each word in comment
                    for i,word in enumerate(comment):
                        if word == target_word:
                            comment[i] = mapped_word
            
            # joining comment words
            document = " ".join(comment)
            document = document.strip()
                        
            processed_corpus.append(document)
            
        return processed_corpus

def get_term_list(path):
        '''
        Function to import term list file
        '''
        word_list = []
        with open(path,"r") as f:
            for line in f:
                word = line.replace("\n","").strip()
                word_list.append(word)
        return word_list

term_badword_list = get_term_list("data_files/badwords_list.txt")

###############################
# Text Preprocessing Pipeline #
###############################
example_text = "I'd not hate you"
example_data = {
    "text" : [example_text]
}
df = pd.DataFrame(example_data)

def text_preprocessing_pipeline(df=df,
                                remove_urls=False,
                                remove_characters=False,
                                reduce_elongated=False,
                                reduce_accented=False,
                                abbreviation_correction=False,
                                normalize_emoticons=False,
                                lower_case=False,
                                normalize_badterm=False,
                                spelling_correction=False,
                                remove_numeric=False,
                                remove_punctuations=False,
                                lemmatization=False
                               ):
    """Preprocess text data in a DataFrame."""

    # Apply preprocessing steps
    print('Text Preprocessing: Developing NER tag count')
    df['ner_tags'] = df['text'].apply(pt.get_ner)
    print('Text Preprocessing: Developing POS tag count')
    df['pos_tags'] = df['text'].apply(pt.get_pos_tag)

    if remove_urls:
        print('Text Preprocessing: Remove urls, user mention, emails')
        df['text_check'] = df['text'].apply(lambda x: pt.remove_urls(x))
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_mention(x))
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_emails(x))

    if remove_characters:
        print('Text Preprocessing: Remove single characters')
        df['text_check'] = df['text_check'].apply(pt.remove_space_single_chars)

    if reduce_elongated:
        print('Text Preprocessing: Reduce elongated characters')
        df['text_check'] = df['text_check'].apply(lambda x: pt.normalize_text_with_original_casing(x))
        
    if reduce_accented:
        print('Text Preprocessing: Reduce accented characters')
        df['text_check'] = df['text_check'].apply(pt.remove_accented_chars)

    if abbreviation_correction:
        print('Text Preprocessing: Expand contraction')
        df['text_check'] = df['text_check'].apply(pt.expand_contractions_with_original_casing)
        print('Text Preprocessing: Correct abbreviation or slang')
        df['text_check'] = df['text_check'].apply(lambda x: pt.slang_resolution__with_original_casing(x))

    if normalize_emoticons:
        print('Text Preprocessing: Normalize emoticons')
        df['text_check'] = df['text_check'].apply(lambda x: pt.convert_emojis(x))
        df['text_check'] = df['text_check'].apply(lambda x: pt.convert_emoticons(x))
        df['text_check'] = df['text_check'].apply(pt.replace_emoticons_with_descriptions)

    if lower_case:
        print('Text Preprocessing: Lowercase')
        df['text_check'] = df['text_check'].str.lower()

    if normalize_badterm:
        print('Text Preprocessing: Replace obfuscated bad term')
        # unique words in vocab 
        unique_words = pt.get_vocab(corpus= df['text_check'])    
        # creating mapping dict 
        mapping_dict = create_profane_mapping(profane_words=term_badword_list,vocabulary=unique_words)
        df['text_check'] = replace_words(corpus=df['text_check'], mapping_dict=mapping_dict)

    if spelling_correction:
        print('Text Preprocessing: Correct spelling')
        df['text_check'] = df['text_check'].apply(lambda x: tool.correct(x))
        df['text_check'] = df['text_check'].apply(pt.expand_contractions_with_original_casing)

    if remove_numeric:
        print('Text Preprocessing: Remove numeric character')
        df['text_check'] = df['text_check'].apply(pt.remove_numeric)

    if remove_punctuations:
        print('Text Preprocessing: Remove punctuations')
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_special_chars(x))
        
    print('Text Preprocessing: Remove multiple spaces')
    df['text_check'] = df['text_check'].apply(lambda x: ' '.join(x.split()))

    print('Text Preprocessing: Tokenisation')
    df["tokenize_text"] = df.apply(lambda row: nltk.word_tokenize(row['text_check'].lower()), axis=1)

    if lemmatization:
        print('Text Preprocessing: Lemmatization')
        df["lemmatized_text"] = df["tokenize_text"].apply(pt.lemmatize_word)
        df['clean_text'] = df['lemmatized_text'].apply(lambda tokens: ' '.join(tokens))
        
    # Remove empty texts
    df = df[~df['clean_text'].isna()]
    df = df[df['clean_text'] != '']
    df = df.reset_index(drop=True)
    
    print('Done')

    return df['clean_text'].tolist()

########################
# Create torch dataset #
########################
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# Define a prediction function for LIME
def predict_for_lime(texts):
    # inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512)

    # Create torch dataset
    input_text_dataset = Dataset(inputs)
    
    # Define test trainer
    pred_trainer = Trainer(model)
    
    # Make prediction using the trainer
    raw_pred, _, _ = pred_trainer.predict(input_text_dataset)
    
    # Make prediction
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(torch.tensor(raw_pred), dim=1).numpy()
    return probabilities
    # with torch.no_grad():
    #     output = model(**inputs)
    # probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
    # return probabilities.cpu().numpy()

# Model Setup
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('thentszeyen/finetuned_cb_detection', num_labels=2)
    return tokenizer, model


# Streamlit user interface components
st.title('Cyberbullying Detection Application')
st.write("This application uses a Transformer model to detect potential cyberbullying in text inputs. Enter text below and press 'Analyze'.")

# Text input from user
with st.spinner("Setting up.."):
    tokenizer, model = load_model()

st.markdown("---")
input_text = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

# Read data 
if input_text and button:
    input_data = {"text" : [input_text]}
    bully_data = pd.DataFrame(input_data)

    with st.spinner("Hold on.. Preprocessing the input text.."):
        cleaned_input_text = text_preprocessing_pipeline(df=bully_data,
                                    remove_urls=True,
                                    remove_characters=True,
                                    reduce_elongated=True,
                                    reduce_accented=True,
                                    abbreviation_correction=True,
                                    normalize_emoticons=True,
                                    lower_case=True,
                                    normalize_badterm=True,
                                    spelling_correction=True,
                                    remove_numeric=True,
                                    remove_punctuations=True,
                                    lemmatization=True
                                )
        
    # Button to trigger model inference
    with st.spinner("Almost there.. Analyzing your input text.."):
            input_text_tokenized = tokenizer(cleaned_input_text, padding=True, truncation=True, max_length=512)

            # Create torch dataset
            input_text_dataset = Dataset(input_text_tokenized)

            # Define test trainer
            pred_trainer = Trainer(model)

            # Make prediction
            raw_pred, _, _ = pred_trainer.predict(input_text_dataset)

            # Preprocess raw predictions
            text_pred = np.where(np.argmax(raw_pred, axis=1)==1,"Cyberbullying Post","Non-cyberbullying Post")

            if text_pred.tolist()[0] == "Non-cyberbullying Post":
                st.success("No worry! Our model says this is a Non-cyberbullying Post!", icon="✅")
            elif text_pred.tolist()[0] == "Cyberbullying Post":
                st.warning("Warning!! Our model says this is a Cyberbullying Post!", icon="⚠️")

            # Generate LIME explanation
            explainer = LimeTextExplainer(class_names=["Non-Cyberbullying", "Cyberbullying"])
            exp = explainer.explain_instance(cleaned_input_text[0], predict_for_lime, num_features=6)
            st.markdown("### Explanation")
            html_data = exp.as_html()
            st.subheader('Lime Explanation')
            components.v1.html(html_data, width=1100, height=350, scrolling=True)

# Footer with additional information or links
st.markdown("---")
st.info("For more information or to report issues, visit our [GitHub repository](https://github.com/ThenTszeYen).")
