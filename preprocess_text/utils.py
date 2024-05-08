# Import Modules/Library
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

nlp = spacy.load('en_core_web_sm')

tool = language_tool_python.LanguageTool('en-US')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_ner(x):
        return " ".join([ent.label_ for ent in nlp(x).ents])

def get_pos_tag(x):
        return " ".join([token.pos_ for token in nlp(x)])

def remove_urls(x):
        return re.sub(r"\b(?:http|https|ftp|ssh)://[^\s]*", '', x)

def remove_mention(x):
        return re.sub(r"@\w+", '', x)

def remove_emails(x):
        return re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", x)

def remove_space_single_chars(x):
        temp = re.sub(r'(?i)(?<=\b[a-z]) (?=[a-z]\b)', '', x)
        return temp
    
def normalize_text(text):
        # Handle specific patterns of laughter
        text = re.sub(r'\b(ha)+\b', 'haha', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(lol)+\b', 'lol', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(lmao)+\b', 'lmao', text, flags=re.IGNORECASE)

        # Reduce elongated sequences of characters
        pattern = re.compile(r"(.)\1{2,}")
        text = pattern.sub(lambda match: match.group(1), text)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = re.sub(r'(.)\1{2,}',r'\1',text)   #any characters, numbers, symbols
        text = re.sub(r'(..)\1{2,}', r'\1', text)  
        text = re.sub(r'(...)\1{2,}', r'\1', text)
        text = re.sub(r'(....)\1{2,}', r'\1', text)  

        return text
    
def normalize_text_with_original_casing(text):
        # Store original casing of words
        original_casing_mapping = {}
        
        # Find unique words and store their original casing
        for word in set(text.split()):
            original_casing_mapping[word.lower()] = word
        
        # Normalize text by converting to lowercase and reducing elongated words
        normalized_text = normalize_text(text.lower())
        
        # Restore original casing using the mapping
        restored_text = " ".join(original_casing_mapping.get(word, word) for word in normalized_text.split())
        
        return restored_text

# Define the function to replace emoticons with descriptions
def replace_emoticons_with_descriptions(text):
    # Load Emoji Dictionary
        emoji_dict_path = 'data_files/Emoji_Dict.p'
        with open(emoji_dict_path, 'rb') as file:
            emoji_dict = pickle.load(file)

        # Load the emoticon dictionary from the JSON file
        emoticon_dict_path = 'data_files/emoticon_dict.json'
        with open(emoticon_dict_path, 'r') as file:
            emoticon_dict = json.load(file)
        
        # Define unwanted characters explicitly
        unwanted_chars = "[£♛™→✔♡†☯♫✌®تح♕★ツ☠♚©♥█║▌│☁☀ღ◄ ▲ ► ▼ ◄ ▲ ► ▼▼ ◄ ▲ ► ▼﻿ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄﻿ ▲ ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄▼﻿ ◄ ▲ ►… — … — ¯¯ … ¯ — … ¯ … ¯ ¯ ¯ … – ¯¯ …… ¯¯ ¯ … ¯ ¯¯ ……¯¯ … ¯ ¯ — … ¯¯– – … ¯ ¯ … ¯ ¯¯ … ¯¯ – … – ¯¯ ¯ — — ¯ ¯¯ – … – ¯¯ — — ¯ … ¯ ¯¯ – ¯¯ … – … —– ¯ …… ¯¯ … ¯ — ¯¯ … … ¯]"
        
        # Remove unwanted characters
        text = re.sub(unwanted_chars, " ", text)

        # Replace emoticons with descriptions
        for emoticon, description in emoticon_dict.items():
            text = text.replace(emoticon, description)

        return text
    
add_emoticon = {'-.-': 'shame',
      '-_-': 'squiting',
      '^.^': 'happy',
      ':0': 'surprise',
      '^-^': 'happy',
      ':33': 'happy face smiley',
      '^__^': 'happy',
      '-____-': 'shame',
      'o_o': 'confused',
      'O_O': 'confused',
      'x3': 'Cute',
      'T T': 'Cry'
      }

EMOTICONS_EMO.update(add_emoticon)

pattern_emoticon = u'|'.join(k.replace('|','\\|') for k in EMOTICONS_EMO)
pattern_emoticon = pattern_emoticon.replace('\\','\\\\')
pattern_emoticon = pattern_emoticon.replace('(','\\(')
pattern_emoticon = pattern_emoticon.replace(')','\\)')
pattern_emoticon = pattern_emoticon.replace('[','\\[')
pattern_emoticon = pattern_emoticon.replace(']','\\]')
pattern_emoticon = pattern_emoticon.replace('*','\\*')
pattern_emoticon = pattern_emoticon.replace('+','\\+')
pattern_emoticon = pattern_emoticon.replace('^','\\^')
pattern_emoticon = pattern_emoticon.replace('·','\\·')
pattern_emoticon = pattern_emoticon.replace('\{','\\{')
pattern_emoticon = pattern_emoticon.replace('\}','\\}')
pattern_emoticon = pattern_emoticon.replace('<','\\>')
pattern_emoticon = pattern_emoticon.replace('>','\\>')
pattern_emoticon = pattern_emoticon.replace('?','\\?')

    # Convert emoticons into word
def convert_emoticons(x):
        for emot in EMOTICONS_EMO:
            x = x.replace(emot, "_".join(EMOTICONS_EMO[emot].replace(",","").replace(":","").split()))
        return x

    # Count emoji
pattern_emoji = u'|'.join(k.replace('|','\\|') for k in UNICODE_EMOJI)
pattern_emoji = pattern_emoji.replace('\\','\\\\')
pattern_emoji = pattern_emoji.replace('(','\\(')
pattern_emoji = pattern_emoji.replace(')','\\)')
pattern_emoji = pattern_emoji.replace('[','\\[')
pattern_emoji = pattern_emoji.replace(']','\\]')
pattern_emoji = pattern_emoji.replace('*','\\*')
pattern_emoji = pattern_emoji.replace('+','\\+')
pattern_emoji = pattern_emoji.replace('^','\\^')
pattern_emoji = pattern_emoji.replace('·','\\·')
pattern_emoji = pattern_emoji.replace('\{','\\{·')
pattern_emoji = pattern_emoji.replace('\}','\\}·')


    # Convert emoji into word
def convert_emojis(x):
        for emot in UNICODE_EMOJI:
            x = x.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
        return x
def get_vocab(corpus):
        '''
        Function returns unique words in document corpus
        '''
        # vocab set
        unique_words = set()
        
        # looping through each document in corpus
        for document in tqdm(corpus):
            for word in document.split(" "):
                if len(word) > 2:
                    unique_words.add(word)
        
        return unique_words

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
    
def remove_accented_chars(x):
        x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return x

def slang_resolution(x):
        slang_path = 'data_files/SLANG_SOCIAL.pkl'
        with open(slang_path, 'rb') as fp:
            slang_path = pickle.load(fp)
        clean_text = []
        for text in x.split():
            if text in list(slang_path.keys()):
                for key in slang_path:
                    value = slang_path[key]
                    if text == key:
                        clean_text.append(text.replace(key,value))
                    else:
                        continue
            else:
                clean_text.append(text)
        return " ".join(clean_text)

    # Sample function to normalize text with original casing
def slang_resolution__with_original_casing(text):
        
        # Store original casing of words
        original_casing_mapping = {}
        
        # Find unique words and store their original casing
        for word in set(text.split()):
            original_casing_mapping[word.lower()] = word
        
        # Normalize text by converting to lowercase and reducing elongated words
        normalized_text = slang_resolution(text.lower())
        
        # Restore original casing using the mapping
        restored_text = " ".join(original_casing_mapping.get(word, word) for word in normalized_text.split())
        
        return restored_text

    # Function to expand contractions in text
def expand_contractions(text):
        # Define the CONTRACTION_MAP dictionary
        CONTRACTION_MAP = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "wont": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
        "stfu": "shut the fuck up",
        "wtf": "what the fuck",
        " u ": " you ",
        " ur ": " your ",
        " n ": " and ",
        " dis ": " this ",
        "'d": " would",
        }
        for contraction, expansion in CONTRACTION_MAP.items():
            text = text.replace(contraction, expansion)
        return text

    # Sample function to normalize text with original casing
def expand_contractions_with_original_casing(text):
        # Store original casing of words
        original_casing_mapping = {}
        
        # Find unique words and store their original casing
        for word in set(text.split()):
            original_casing_mapping[word.lower()] = word
        
        # Normalize text by converting to lowercase and expanding contractions
        normalized_text = expand_contractions(text.lower())
        
        # Restore original casing using the mapping
        restored_text = " ".join(original_casing_mapping.get(word, word) for word in normalized_text.split())
        
        return restored_text

def remove_numeric(x):
        return ''.join([i for i in x if not i.isdigit()])

def remove_special_chars(x):
        punct = string.punctuation + "¶“”‘’" 
        for p in punct:
            x = x.replace(p, " ")
        return x

def lemmatize_word(text):
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word) for word in text]
        return lemmas
    
