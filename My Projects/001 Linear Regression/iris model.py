
import seaborn as sns
import pandas as pd
import numpy as np

df=sns.load_dataset('iris')
print(df.columns)
print(df.head(2))
# 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['species']=le.fit_transform(df['species'])
# selecting x and y
x=df.drop('species',axis=1)
y=df.species

# train test split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)

# model fitting
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)

#  model prediction
print(model.predict(xtest[0:1]))
print(model.score(xtrain,ytrain))

#  saving model
import joblib
from joblib import parallel,delayed
# Save the model as a pickle in a file
joblib.dump(model, 'model2.pkl')
  
# Load the model from the file
model2 = joblib.load('model2.pkl')
  
# Use the loaded model to make predictions
# model2.predict(X_test)

import pickle
import joblib
import streamlit as st
from PIL import Image

# Load the model from the file
model2 = joblib.load('model2.pkl')

st.title('Iris Flower Prediction')
html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html = True)

sepal_length = st.text_input("Sepal Length", "Type Here")
sepal_width = st.text_input("Sepal Width", "Type Here")
petal_length = st.text_input("Petal Length", "Type Here")
petal_width = st.text_input("Petal Width", "Type Here")
result=""
if st.button("Predict"):
    result = model2.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.success('The output is {}'.format(result))



