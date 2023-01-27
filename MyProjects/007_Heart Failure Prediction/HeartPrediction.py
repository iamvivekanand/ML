import streamlit as st 
import joblib
model=joblib.load('HeartFailureModel.pkl')
st.success('This App has been deployed by Vivekanand')

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

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

pred=model.predict([[age,anaemia,cp,diabetes,ef,hbp,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])

if st.button('Predict'):
    if pred==0:
        st.success('No failure')
    else:
        st.warning('Heart Failure')

