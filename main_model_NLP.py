import pickle
import numpy as np
import pandas as pd

dataset = pd.read_csv("train.csv",encoding='latin-1')
x = dataset.iloc[:,2].values
y = dataset.iloc[:,1].values

import nltk,re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
cleaned_tweets_list = []
for i in range(0, len(x)):
    tweets_cleaned = re.sub('http[s]?://\S+', '', x[i])
    tweets_cleaned = re.sub('[^a-zA-Z]', ' ',tweets_cleaned)
    tweets_cleaned = tweets_cleaned.lower()
    tweets_cleaned = tweets_cleaned.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    tweets_cleaned = [ps.stem(word) for word in tweets_cleaned if not word in set(all_stopwords)]
    tweets_cleaned = ' '.join(tweets_cleaned)
    cleaned_tweets_list.append(tweets_cleaned)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=980)
x = cv.fit(cleaned_tweets_list).toarray()

vectorizer = pickle.dump(cv,open("vectorizer.pkl",'wb'))


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

pickle.dump(classifier,open("model.pkl",'wb'))
model = pickle.load(open("model.pkl",'rb'))


# y_pred = classifier.predict(x_test)
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,y_pred))
