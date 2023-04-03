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
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
import func as fc 
from io import StringIO
st.set_page_config(layout='wide')  
tab1, tab2, tab3 = st.tabs(['Data','Visualization','ML'])
optionList = ['Gender and Churn Distribution','Customer Contract Distribution','Payment Method Distribution','Payment Method Distribution Churn',
              'Churn Distribution w.r.t Internet Service and Gender','Dependents Distribution Churn',
              'Churn Distribution w.r.t Partners','Churn Distribution w.r.t Senior Citizens',
              'Churn Distribution w.r.t Online Security','Churn Distribution w.r.t Paperless Billing',
              'Churn Distribution w.r.t Tech Support','Churn Distribution w.r.t Phone Service',
              'Tenure vs. Churn']
#option to upload the dataframe
with tab1: 
    option = st.selectbox('Select the plot you want to visualize',optionList)
    uploaded_dataframe = st.file_uploader("Choose a file")
if uploaded_dataframe is not None: 
    if option is not None : 
        figure = fc.take_input(uploaded_dataframe,option)
        with tab2: 
            st.plotly_chart(figure,use_container_width=True)  