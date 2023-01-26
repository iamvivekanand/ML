# importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import joblib
model=joblib.load('carprice.pkl')

st.title('Car price Prediction Model')
symboling=st.number_input("Enter Symboling : ",min_value=-2,max_value=3,format='%f')
wheelbase=st.number_input("Enter Wheel Base",min_value=85.0,max_value=121.0,format='%f')
carlength=st.number_input("Enter Car Length: ",min_value=140,max_value=210,format='%f')
carwidth=st.number_input("Enter Car Width: ",min_value=60,max_value=73,format='%f')
carheight=st.number_input("Enter Car Height: ",min_value=45,max_value=70,format='%f')
curbweight=st.number_input("Enter Curbweight: ",min_value=1200,max_value=5000,format='%f')
enginesize=st.number_input("Enter Engine Size: ",min_value=60,max_value=350,format='%f')
boreratio=st.number_input("Enter Bore Ratio: ",min_value=2,max_value=4,format='%f')
stroke=st.number_input("Enter Stroke Value: ",min_value=2,max_value=5,format='%f')
compressionratio=st.number_input("Enter Compression Ratio: ",min_value=7,max_value=23,format='%f')
horsepower=st.number_input("Enter Horse Power: ",min_value=48,max_value=278,format='%f')
peakrpm=st.number_input("Enter Peak RPM: ",min_value=3000,max_value=7000,format='%f')
citympg=st.number_input("Enter City MPG: ",min_value=10,max_value=60,format='%f')
highwaympg=st.number_input("Enter Highway MPG: ",min_value=10,max_value=60,format='%f')

button=st.button('Calculate Price')
predicted=1
if button:
    predicted=model.predict([['symboling','wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg']])
st.write(predicted)

