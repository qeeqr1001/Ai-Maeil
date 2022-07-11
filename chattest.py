import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer 

stemmer=PorterStemmer() #형태소분석기 생성

def tokenize(sentence): #토큰화
    return nltk.word_tokenize(sentence)

def stem(word): #형태소분석
    return stemmer.stem(word.lower()) #lower=소문자로 바꾸는

def bag_of_words(tokenized_sentence, all_words): #토큰화된 문장과 데이터의 모든 단어 비교하여 일치하면 숫자가 추가되는식?
    """
    sentence=["hello","how","are","you"]
    words=["hi","hello","I","you","bye","thank","cool"]
    bog =[  0  ,   1  ,  0 ,  1  ,  0  ,   0  ,    0]
    """   
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words): #idx=index
        if w in tokenized_sentence:
            bag[idx] =1.0
    return bag

sentence=["hello","how","are","you"]
words=["hi","hello","I","you","bye","thank","cool"]
bog=bag_of_words(sentence,words)
print(bog)

# words=["organize","organizes","organizing"]
# stemmed_words=[stem(w) for w in words]
# print(stemmed_words)

# a="How long does shipping take?"
# print(a)
# a=tokenize(a) 
# print(a)
