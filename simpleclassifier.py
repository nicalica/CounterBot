import pickle
import re
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

class target_classifier:
    def __init__(self, vectorizer_path='Models/Embeddings/tfidf_vectorizer_nostop',
                 model_path='Models/MultiClassifier/LR_tfidf_matrix_nostop'):
        with open(vectorizer_path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            self.lr_model = pickle.load(f)

    def utils_preprocess_text(self, message: str, flg_stemm=False, flg_lemm=False, lst_stopwords=None) -> str:
        '''
        Code from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
        Preprocess a string.
        :parameter
            :param message: string - text to process
            :param lst_stopwords: list - list of stopwords to remove
            :param flg_stemm: bool - whether stemming is to be applied
            :param flg_lemm: bool - whether lemmitization is to be applied
        :return
            cleaned text
        '''

        # clean (convert to lowercase and remove punctuations and
        # characters and then strip)
        message = re.sub(r'[^\w\s]', '', str(message).lower().strip())

        # Tokenize (convert from string to list)
        lst_text = message.split()

        # remove Stopwords
        if lst_stopwords is not None:
            lst_text = [word for word in lst_text if word not in
                        lst_stopwords]

        # Stemming (remove -ing, -ly, ...)
        if flg_stemm == True:
            ps = PorterStemmer()
            lst_text = [ps.stem(word) for word in lst_text]

        # Lemmatization (convert the word into root word)
        if flg_lemm == True:
            lem = WordNetLemmatizer()
            lst_text = [lem.lemmatize(word) for word in lst_text]

        # back to string from list
        message = " ".join(lst_text)
        return message

    def classify(self, texts: List[str]):

        texts = [self.utils_preprocess_text(text) for text in texts]
        vectorized_texts = self.tfidf_vectorizer.transform(texts)

        # has probabilities for each class
        probabilities = self.lr_model.predict_proba(vectorized_texts)

        # return classes with highest probs for each text
        predicted_classes = [0] * len(texts)
        predicted_probs = [0] * len(texts)

        target_types = ['Disabled', 'Jews', 'LGBT+', 'Migrants', 'Muslims', 'POC', 'Women', 'Other/Mixed', 'None']

        for i, probs in enumerate(probabilities):
            max_prob = max(probs)
            max_class = np.where(probs == max_prob)[0][0]
            predicted_classes[i] = target_types[max_class]
            predicted_probs[i] = max_prob

        return predicted_classes, predicted_probs
