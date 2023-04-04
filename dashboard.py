import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots 
import warnings
import streamlit as st 
warnings.filterwarnings('ignore') 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
import func as fc 
from io import StringIO
st.set_page_config(layout='wide')  
tab1, tab2, tab3 = st.tabs(['Data','Visualization','ML'])
#loading the options list from the functions file func.py 
optionList = fc.OPTION_LIST
modelList = fc.MODEL_SELECTOR
#option to upload the dataframe
with tab1: 
    option = st.selectbox('Select the plot you want to visualize',optionList)
    uploaded_dataframe = st.file_uploader("Choose a file")
    #print(type(uploaded_dataframe))
if uploaded_dataframe is not None: 
    if option is not None : 
        figure,processed_df = fc.take_input(uploaded_dataframe,option)
        with tab1: 
            st.dataframe(processed_df)
        with tab2: 
            st.plotly_chart(figure,use_container_width=True)  
with tab3: 
    modeloption = st.selectbox('Select an ML Model',modelList)
    uploaded_dataframe = st.file_uploader("Choose a file", key=2)
    test_size_slider = st.slider('Enter the test size: ',0.0,1.0)
    random_state_input = st.number_input('Select a random seed',0,1000)
    #print(test_size_slider)
    if uploaded_dataframe is not None:    
        #Add a slider later the test_size, and a input box for the random state
        #print(uploaded_dataframe)
        acc_score, classification_rep, output_df,original_df = fc.standardize_dataframe(uploaded_dataframe,modeloption,test_size_slider,random_state_input) 
        st.dataframe(output_df)
        #st.write('Accuracy Score of '+modeloption+' is: '+str(acc_score))
        st.metric(label='Accuracy Score of '+modeloption,value=str(acc_score))
        st.markdown('```bash \t \n'+classification_rep+'```')
        #print(acc_score,'\n',classification_rep)
        st.write('Enter some information to predict the churn:')
        pr_1 = st.selectbox('Select the gender:',['Female','Male'])
        pr_2 = st.selectbox('Is the customer a senior citizen?',['Yes','No'])
        pr_3 = st.selectbox('Does the customer have a partner?',['Yes','No']) 
        pr_4 = st.selectbox('Does the customer have dependents?',['Yes','No']) 
        pr_5 = st.number_input('What is the customer tenure?',0,100)
        pr_6 = st.selectbox('Does the customer have phone service?',['Yes','No']) 
        pr_7 = st.selectbox('Does the customer have multiple lines?',['Yes','No','No phone service'])
        pr_8 = st.selectbox('Does the customer have internet service?',['No','DSL','Fiber optic'])
        pr_9 = st.selectbox('Does the customer have online security?',['Yes','No','No internet service'])
        pr_10 = st.selectbox('Does the customer have online backup?',['Yes','No','No internet service'])
        pr_11 = st.selectbox('Does the customer have device protection?',['Yes','No','No internet service'])
        pr_12 = st.selectbox('Does the customer have tech support?',['Yes','No','No internet service'])
        pr_13 = st.selectbox('Does the customer have streaming TV?',['Yes','No','No internet service'])
        pr_14 = st.selectbox('Does the customer have streaming movies?',['Yes','No','No internet service'])
        pr_15 = st.selectbox('Does the customer have a contract?',['Month-to-month','One year']) 
        pr_16 = st.selectbox('Does the customer have paperless billing?',['Yes','No']) 
        pr_17 = st.selectbox('What is the payment method of the customer?',['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
        pr_18 = st.number_input('What are the monthly charges of the customer?')
        pr_19 = st.number_input('What are the total charges of the customer?')
        if st.button('Predict Churn'): 
            #convert the inputs to a vector and pass it to a voting classifier algorithm
            feature_vector = pd.DataFrame({'customerID':[1],
                                           'gender':[pr_1], 
                                          'SeniorCitizen':[pr_2],
                                          'Partner':[pr_3],
                                          'Dependents':[pr_4],
                                          'tenure':[pr_5],
                                          'PhoneService':[pr_6],
                                          'MultipleLines':[pr_7],
                                          'InternetService':[pr_8],
                                          'OnlineSecurity':[pr_9],
                                          'OnlineBackup':[pr_10],
                                          'DeviceProtection':[pr_11],
                                          'TechSupport':[pr_12], 
                                          'StreamingTV':[pr_13],
                                          'StreamingMovies':[pr_14],
                                          'Contract':[pr_15],
                                          'PaperlessBilling':[pr_16],
                                          'PaymentMethod':[pr_17],
                                          'MonthlyCharges':[pr_18],
                                          'TotalCharges':[pr_19]})
            #passing the feature vector to be processed and predict a churn output 
            #print(feature_vector)
            response = fc.standardize_feature_vector(feature_vector,original_df,test_size_slider,random_state_input)
            st.metric(label='Prediction Response',value=response)
