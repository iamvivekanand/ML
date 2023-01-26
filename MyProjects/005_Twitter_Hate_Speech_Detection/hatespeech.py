from nltk.util import pr
# Pretty print a sequence of data items
import re
import nltk
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
# CountVectorizer Converts a collection of text documents to a matrix of token counts
from sklearn.model_selection import train_test_split 
stemmer=nltk.SnowballStemmer('english')
stopword=set(stopwords.words('english'))
data = pd.read_csv("twitter.csv")
del data['Unnamed: 0']
# Now we will work over "tweet: and "labels" columns only for our detection model
data['labels']=data['class'].map({0:"hate speech",1: "Offensive Language", 2: "No Hate and Offensive"})
data=data[['tweet','labels']]
# data cleaning
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
# selecting dependent and independent variables
X=np.array(data['tweet'])
y=np.array(data['labels'])
# 
cv=CountVectorizer()
X=cv.fit_transform(X)
# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=41)
# 
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
# 
model.fit(X_train,y_train)
# 
import streamlit as st
import streamlit as st
# st.markdown("<body style='background-color:powderblue;'>")
st.markdown("<h1 style='color: blue;'>Twitter Hate Speech Detection</h1>", unsafe_allow_html=True)
# st.title('Twitter Hate Speech Detection')
user_text=st.text_input('Enter something here: ')
click=st.button('Predict')
res=''
if click:
    user_text=cv.transform([user_text]).toarray()
    res=model.predict(user_text)
st.success(res)
