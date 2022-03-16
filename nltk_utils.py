import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import numpy as np


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1

    return bag

    

# testing tokenize & stem (succeed)
#a = "I hate my job but I am scared to quit and seek a new career."
#print(a)
#a = tokenize(a)
#print(a)

#words = ["organize", "organizes", "organizing"]
#stemmed_words = [stem(w) for w in words]
#print(stemmed_words)
