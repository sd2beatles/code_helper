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

