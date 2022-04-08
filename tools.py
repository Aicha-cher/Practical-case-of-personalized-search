
import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])
install(spacy)
install(contractions)
install(gensim)
import spacy
import contractions
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader
import numpy as np

def read_data (path):
    with open(path) as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
    return lines

def text_constructor(lines, query:bool=None):
    _dict = {}
    _id = ""
    doc_text = ""
    for l in lines:
        if l.startswith(".I"):
            _id = int(l.split(" ")[1].strip())
        elif not query :
            if l.startswith(".X"):
                _dict[_id] = doc_text.lstrip(" ")
                _id = ""
                doc_text = ""
            else:
                doc_text += l.strip()[3:] + " " # The first 3 characters of a line can be ignored.
        else:
            if l.startswith(".W"):
                _dict[_id] = l.strip()[3:]
                _id = ""


    # Print something to see the dictionary structure, etc.
    print(f"Number of documents = {len(_dict)}" + ".\n")
    return _dict

def data_cleaning(dict):
    cleaned_doc_df = pd.DataFrame()
    nlp = spacy.load('en_core_web_sm',disable=['ner','parser'])
    nlp.max_length=5000000
    cleaned_doc_df['raw'] = pd.DataFrame(dict.values())
    #lowercase
    cleaned_doc_dict = {k:para.lower() for k,para in dict.items()}
    # Expand Contractions
    cleaned_doc_dict = {k:contractions.fix(para) for k,para in cleaned_doc_dict.items()}
    cleaned_doc_df['cleaned'] = pd.DataFrame(cleaned_doc_dict.values())

    # Stopwords removal & Lemmatizing tokens using SpaCy
    cleaned_doc_df['lemmatized']= cleaned_doc_df['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (not token.is_stop and not token.is_punct)]))
    cleaned_doc_df['lemmatized']= cleaned_doc_df['lemmatized'].apply(lambda x: re.sub(r"[\.\,\#_\|\:\?\?\/\=]", ' ',x))
    cleaned_doc_df['lemmatized']= cleaned_doc_df['lemmatized'].apply(lambda x: re.sub(' +',' ',x))
    #document number column
    cleaned_doc_df["docno"] = pd.factorize(cleaned_doc_df['lemmatized'])[0].astype(str)
    return cleaned_doc_df

def get_embedding_w2v(doc_tokens, w2vec):
    embeddings = []
    if len(doc_tokens)<1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2vec.index_to_key:
                embeddings.append(w2vec.get_vector(tok))
            else:
                embeddings.append(np.random.rand(300))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)

def vectorizor(cleaned_df):
    #method 1
    tfIdfVectorizer=TfidfVectorizer()
    tfIdf_vec = tfIdfVectorizer.fit_transform(cleaned_df['lemmatized'])
    #adding tf-idf to the dataframe
    cleaned_df['TF-IDF'] = pd.Series([ j for j in tfIdf_vec.toarray()])
    # method 2
    w2vec = gensim.downloader.load('word2vec-google-news-300')
    cleaned_df['w2v_vector']=cleaned_df['lemmatized'].apply(lambda x :get_embedding_w2v(x.split(), w2vec))

    return cleaned_df