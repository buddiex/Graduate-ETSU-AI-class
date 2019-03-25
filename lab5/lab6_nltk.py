import nltk

# nltk.download()
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import ngrams
from bs4 import BeautifulSoup
import urllib.request


def frequency_analysis(tokens):
    print("Frequency Analysis...")
    freq = nltk.FreqDist(tokens)  # lowercase, non-punctuated tokens
    for key, val in freq.items():
        print(str(key) + ':' + str(val))
    print("Length of Unique Items:", len(freq.items()))
    freq.plot(20, cumulative=False)


print("Reading Text and Tokenizing...")
response = urllib.request.urlopen('https://www.gutenberg.org/files/74/74-h/74-h.htm')
html = response.read()
soup = BeautifulSoup(html, "html5lib")
text = soup.get_text(strip=True)
s_tokens = sent_tokenize(text)
w_tokens = word_tokenize(text)
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)
tokens = [t.lower() for t in tokens]

frequency_analysis(tokens)
print("Removing Stop Words...")
clean_tokens = tokens[:]
sr = stopwords.words('english')
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)

frequency_analysis(clean_tokens)

print("Stemming...")
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in clean_tokens]
print("Frequency Analysis...")
frequency_analysis(stemmed_tokens)

print("Lemmatizing...")
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in clean_tokens]
print("Frequency Analysis...")
frequency_analysis(lemmatized_tokens)


print("POS Analysis...")
import matplotlib.pyplot as plt
pos = nltk.pos_tag(lemmatized_tokens)
pos_counts = {}
for key,val in pos:
    print(str(key) + ':' + str(val))
    if val not in pos_counts.keys():
        pos_counts[val] = 1
    else:
        pos_counts[val] += 1
print(pos_counts)
plt.bar(range(len(pos_counts)), list(pos_counts.values()), align='center')
plt.xticks(range(len(pos_counts)), list(pos_counts.keys()))
plt.show()

print("Tri-Grams...")
trigrams = ngrams(text.split(), 3)
for gram in trigrams:
    print(gram)

print("Document-Term Matrix...")
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
response = urllib.request.urlopen('https://www.gutenberg.org/files/32325/32325-h/32325-h.htm')
html = response.read()
soup = BeautifulSoup(html,"html5lib")
text2 = soup.get_text(strip=True)
docs = [text, text2]
vec = CountVectorizer()
X = vec.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print("Instances of 'Huck' in both documents:")
print(df["huck"]) # Show the count for this word in both documents
print("Instances of 'Tom' in both documents:")
print(df["tom"]) # Show the count for this word in both documents
print(df) # Show the full data frame