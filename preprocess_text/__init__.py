from preprocess_text import utils

def get_ner(x):
        return utils.get_ner(x)

def get_pos_tag(x):
        return utils.get_pos_tag(x)

def remove_urls(x):
        return utils.remove_urls(x)

def remove_mention(x):
        return utils.remove_mention(x)

def remove_emails(x):
        return utils.remove_emails(x)

def remove_space_single_chars(x):
        return utils.remove_space_single_chars(x)
    
def normalize_text(text):
        return utils.normalize_text(text)
    
def normalize_text_with_original_casing(text):
        return utils.normalize_text_with_original_casing(text)

# Define the function to replace emoticons with descriptions
def replace_emoticons_with_descriptions(text):
        return utils.replace_emoticons_with_descriptions(text)

def convert_emoticons(x):
        return utils.convert_emoticons(x)

def convert_emojis(x):
        return utils.convert_emojis(x)

def get_vocab(corpus):
        return utils.get_vocab(corpus)
    
def remove_accented_chars(x):
        return utils.remove_accented_chars(x)

def slang_resolution(x):
        return utils.slang_resolution(x)

    # Sample function to normalize text with original casing
def slang_resolution__with_original_casing(text):
        return utils.slang_resolution__with_original_casing(text)

    # Function to expand contractions in text
def expand_contractions(text):
        return utils.expand_contractions(text)

    # Sample function to normalize text with original casing
def expand_contractions_with_original_casing(text):
        return utils.expand_contractions_with_original_casing(text)

def remove_numeric(x):
        return utils.remove_numeric(x)

def remove_special_chars(x):
        return utils.remove_special_chars(x)

def lemmatize_word(text):
        return utils.lemmatize_word(text)