import streamlit as st 
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing
# importing dataset 
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
# selecting 'input' and 'target' variables
X=df.drop(['DEATH_EVENT'],axis=1)
y=df['DEATH_EVENT']
# Splitting dataset into training and testing 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# feature scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,
                            criterion='gini',
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features='auto',
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=None,
                            random_state=None,
                            verbose=0,
                            warm_start=False,
                            class_weight=None,
                            ccp_alpha=0.0,
                            max_samples=None)
rf.fit(X_train,y_train)



# model=joblib.load('HeartFailure.pkl')
st.success('This App has been deployed by Vivekanand')

st.title('Heat Failure Prediction')
age=st.number_input('Enter Your Age Below', min_value=35,max_value=100,step=1)
anaemia=st.number_input('if Anaemia:Enter 0 for No, 1 for Yes:', min_value=0,max_value=1,step=1)
cp=st.number_input('Enter value of creatinine phosphokinase ', min_value=20,max_value=8000)
diabetes=st.number_input('If Diabetes:Enter 0 for No, 1 for Yes', min_value=0,max_value=1)
ef=st.number_input('Enter ejection fraction value', min_value=10,max_value=80,step=1)
hbp=st.number_input('High Blood Pressure:0 for No, 1 for yes', min_value=0,max_value=1,step=1)
# age=st.number_input('Enter your age here', min_value=1,max_value=120,step=1)
platelets=st.number_input('Enter platelets count', min_value=25000,max_value=850000)
serum_creatinine=st.number_input('Enter value of Serum Cretinine:', min_value=0,max_value=10)
serum_sodium=st.number_input('Enter sodium level in blood', min_value=100,max_value=150,step=1)
sex=st.number_input('Enter 0 for Female,1 for Male', min_value=0,max_value=1,step=1)
smoking=st.number_input('Smkoing:Enter 0 for No,1 for Yes', min_value=1,max_value=120,step=1)
time=st.number_input('Enter time', min_value=1,max_value=300,step=1)

pred=rf.predict([[age,anaemia,cp,diabetes,ef,hbp,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])

if st.button('Predict'):
    if pred==0:
        st.success('No failure')
    else:
        st.warning('Heart Failure')

