import streamlit as st
import pandas as pd

st.write("""

# 890 Recommender

""")
df = pd.read_csv('Dummy for 890 Rec Sys.csv')
df['text']=df['Title']+' '+df['Description']+' '+df['Industry']
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 4
MAX_WORDS = 200

PATTERN_S = re.compile("\'s")   
PATTERN_RN = re.compile("\\r\\n") 
PATTERN_PUNC = re.compile(r"[^\w\s]") 

def clean_text(text):
    """
    Series of cleaning. String to lower case, remove non words characters and numbers.
        text (str): input text
    return (str): modified initial text
    """
    #text = text.lower() 
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    text = re.sub(PATTERN_PUNC, ' ', text)
    return text

def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
    """
    Lemmatize, tokenize, crop and remove stop words.
    """
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
    else:
        tokens = [w for w in word_tokenize(sentence)]
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                                        and w not in stopwords)]
    return tokens    


def clean_sentences(df):
    """
    Remove irrelavant characters (in new column clean_sentence).
    Lemmatize, tokenize words into list of words (in new column tok_lem_sentence).
    """
    #print('Cleaning sentences...')
    df['text'] = df['text'].apply(clean_text)
    df['n_text'] = df['text'].apply(
        lambda x: tokenizer(x, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True))
    return df
    
df = clean_sentences(df)
def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]  
    return best_index

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Adapt stop words
token_stop = tokenizer(' '.join(STOPWORDS), lemmatize=False)

# Fit TFIDF
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer) 
tfidf_mat = vectorizer.fit_transform(df['text'].values) # -> (num_sentences, num_vocabulary)

def get_recommendations_tfidf(sentence, tfidf_mat):
    """
    Return the database sentences in order of highest cosine similarity relatively to each 
    token of the target sentence. 
    """
     #Embed the query sentence
    tokens = [str(tok) for tok in tokenizer(sentence)]
    vec = vectorizer.transform(tokens)
     #Create list with similarity between query and dataset
    mat = cosine_similarity(vec, tfidf_mat)
     #Best cosine distance for each token independantly
    print(mat.shape)
    best_index = extract_best_indices(mat, topk=10)
    return best_index

text_input = st.text_input(
        "Enter anything"
    )

if text_input:
    best_index = get_recommendations_tfidf(text_input, tfidf_mat)
    st.subheader('Recommendation')
    st.write(df[['Title', 'Description', 'Industry', 'Type of Solution']].iloc[best_index])