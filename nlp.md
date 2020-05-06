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

### 2. Tokenizer (Sentencepieces)


```python
# In the case where the input file csv
with open(input_file, 'a+', encoding='utf-8') as f:
    for idx, row in df.iterrows():
        f.write('{}\n'.format(row['text']))

#
templates= '--input={} \
--pad_id={} \
--bos_id={} \
--eos_id={} \
--unk_id={} \
--model_prefix={} \
--vocab_size={} \
--character_coverage={} \
--model_type={} \
--hard_vocab_limit=false'



train_input_file = "파일 폴더/dialogues_train.txt"
pad_id=0  #<pad> token을 0으로 설정
vocab_size = 2000 # vocab 사이즈
prefix = 'botchan_spm' # 저장될 tokenizer 모델에 붙는 이름
bos_id=1 #<start> token을 1으로 설정
eos_id=2 #<end> token을 2으로 설정
unk_id=3 #<unknown> token을 3으로 설정
character_coverage = 1.0 # to reduce character set 
model_type ='word' # Choose from unigram (default), bpe, char, or word


cmd = templates.format(r'file directory\news.txt',
                pad_id,
                bos_id,
                eos_id,
                unk_id,
                prefix,
                vocab_size,
                character_coverage,
                model_type)


```


