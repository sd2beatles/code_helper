# 0. Regular Expression

![image](https://user-images.githubusercontent.com/53164959/81068099-c6799580-8f1a-11ea-897d-e17e8380d24e.png)



# 1. Natural Laguage Processing
## 1.1 English(nltk)

### 1.1.1 Tokenize
- libraries
```python
from nltk.tokenize import WordPunctTokenizer,word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer

sentences=sent_tokenize(text)
results=list()
for sentence in sentences:
    sentence=re.sub(r'\W+',' ',sentence)
    sentence=re.sub(r'\b\w{1,3}\b',' ',sentence)
    sentence=re.sub('[ ]+',' ',sentence)
    tokens=[lemma.lemmatize(token,'v') for token in word_tokenize(sentence)]
    results.append(tokens)
```
### 1.1.2 StopWords
```python
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))

```

### 1.1.3 vocab_freq(normal code)
```python
#create a vocab_freq 
vocab_freq=defaultdict(int)

#count the frequencies of every word in each sentence
for token in tokens:
            if token not in vocab_dict:
                vocab_freq[token]=1
            else:
                vocab_freq[token]+=1
                
 #sort the vocab_freq by the number of frequencies in descending order
vocb_freq=sorted(voca_freq.items(),key=lambda x:x[1],reverse=True)
```
### 1.1.4 voca_freq(Counter)


```python
#Right after tokenizing all the sentences, we should put them into the list called sentences 
#Concate all the lists into one dimenions
vocabs=np.hstack(sentences)

from collections import Counter
#Using Counter library to create vocab_freq
vocab_freq=Counter(vocabs)
vocab_freq=sorted(vocab_freq.items(),key=lambda x:x[1],reverse=True)
```
### 1.1.5 vocab_freq(nltk)
```python
from nltk import FreqDist
import numpy as np

#convert the lists into one dimension
tokens=np.hstack(sentences)

voca_freq=FreqDist(tokens)
#Limit the number of most common words up to the specific number
voca_most=voca_freq.most_common(10)

```

### 1.1.6 vocab_freq(keras)


