import os
import nltk
import string
import json
import re
import numpy as np
from argparse import ArgumentParser
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="input",
                    help="input dataset in json format", metavar="FILE")
parser.add_argument("-faq", "--faq", dest="faq",
                    help="FAQ document in json format", metavar="FILE")

args = parser.parse_args()

documents = []

with open(args.faq, "r") as read_file:
    faq = json.load(read_file)

for elem in faq:
    text = elem['question'] + elem['answer']
    documents.append(text)

def tokenize_and_stem(s):
            REMOVE_PUNCTUATION_TABLE = str.maketrans({x: None for x in string.punctuation})
            TOKENIZER = TreebankWordTokenizer()
            STEMMER = PorterStemmer()
            return [STEMMER.stem(t) for t 
                in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]

# pre-process stop words    

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

STEMMER = PorterStemmer()
TOKENIZER = TreebankWordTokenizer()

stopwords_stemmed = []
for word in stopwords:
    for token in TOKENIZER.tokenize(word):
        stopwords_stemmed.append(STEMMER.stem(token))

# build tfdif matrix

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.05, stop_words = stopwords_stemmed,
use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = vectorizer.fit_transform(documents)

# score input file using cos similariy


with open(args.input, "r") as read_file:
    input = json.load(read_file)

count = 0
for elem in input:
    query_vector = vectorizer.transform([elem['content']]) 
    similarity = cosine_similarity(query_vector, tfidf_matrix)
    scores = similarity.ravel()
    maxPos = np.argmax(scores)
    bestFAQ = documents[maxPos]
    if np.max(scores)>0.4:
        count = count + 1
        print(elem['content'] + "\nFAQ:" + bestFAQ + "\nScore: " + str(np.max(scores)))
print()
print(str(count) + " Questions answered via FAQ")