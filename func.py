import pandas as pd 
import numpy as np 
from plotly.subplots import make_subplots 
import plotly.graph_objects as go 
import matplotlib.pyplot as plt 
import plotly.express as px 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
OPTION_LIST = ['Gender and Churn Distribution','Customer Contract Distribution','Payment Method Distribution','Payment Method Distribution Churn',
              'Churn Distribution w.r.t Internet Service and Gender','Dependents Distribution Churn',
              'Churn Distribution w.r.t Partners','Churn Distribution w.r.t Senior Citizens',
              'Churn Distribution w.r.t Online Security','Churn Distribution w.r.t Paperless Billing',
              'Churn Distribution w.r.t Tech Support','Churn Distribution w.r.t Phone Service',
              'Tenure vs. Churn']
MODEL_SELECTOR = ['KNN','SVC','RF','LR','DT','Adaboost','Gradient Boosting','Voting Classifier']
num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
scaler= StandardScaler()
def preprocess(df): 
    df = df.drop(['customerID'], axis = 1)
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    df[np.isnan(df['TotalCharges'])]
    df[df['tenure'] == 0].index
    df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
    df[df['tenure'] == 0].index
    df.fillna(df["TotalCharges"].mean())
    df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df 
def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series
def evaluate_voter(test_feature_vector, df,test_size,random_state): 
    print(df)
    df = preprocess(df)
    df = df.apply(lambda x: object_to_int(x))
    X = df.drop(columns = ['Churn'])
    y = df['Churn'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state = random_state, stratify=y)
    df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')),columns=num_cols)
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    clf1 = GradientBoostingClassifier()
    clf2 = LogisticRegression()
    clf3 = AdaBoostClassifier()
    eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
    eclf1.fit(X_train, y_train)
    #feeding the feature vector as a test input 
    predicted_y = eclf1.predict(test_feature_vector)
    if predicted_y[0] == 1: 
        #print('The customer is likely to stop using the services')
        return 'Customer is likely to stop using the telecom services'
    else: 
        #print('The customer is likely to continue using the services')
        return 'Customer is likely to continue using the telecom services'


def standardize_feature_vector(df,original_df, test_size,random_state): 
    df = df.drop(['customerID'], axis = 1) 
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    #Manual label encoding is the only solution here... 
    df["SeniorCitizen"]= df["SeniorCitizen"].map({"No": 0, "Yes": 1})
    df['gender'] = df['gender'].map({'Female':0,'Male':1}) 
    df['Partner'] = df['Partner'].map({"No":0,"Yes":1})
    df['Dependents'] = df['Dependents'].map({"No":0,"Yes":1}) 
    df['PhoneService'] = df['PhoneService'].map({"No":0,"Yes":1}) 
    df['MultipleLines'] = df['MultipleLines'].map({"No phone service":1,"No":0,"Yes":2})
    df['InternetService'] = df['InternetService'].map({'DSL':0,'Fiber optic':1,'No':2}) 
    df['OnlineSecurity'] = df['OnlineSecurity'].map({'No':0,'Yes':2,'No internet service':1}) 
    df['OnlineBackup'] = df['OnlineBackup'].map({'No':0,'Yes':2,'No internet service':1})
    df['DeviceProtection'] = df['DeviceProtection'].map({'No':0,'Yes':2,'No internet service':1})
    df['TechSupport'] = df['TechSupport'].map({'No':0,'Yes':2,'No internet service':1})
    df['StreamingTV'] = df['StreamingTV'].map({'No':0,'Yes':2,'No internet service':1})
    df['StreamingMovies'] = df['StreamingMovies'].map({'No':0,'Yes':2,'No internet service':1})
    df['Contract'] = df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({"No":0,"Yes":1})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check':2, 'Mailed check':3,'Bank transfer (automatic)':0,'Credit card (automatic)':1})
    #Churn -> No:0, Yes:1 
    numpy_vector = df.to_numpy() 
    print(df)
    print(numpy_vector)
    #passing the vector as a test vector to a trained voting classifier
    return evaluate_voter(df,original_df,test_size,random_state)


def standardize_dataframe(filepath,option,test_size,random_state): 
    df = pd.read_csv(filepath)
    #print(df)
    df_new = preprocess(df)
    #print(df)
    #label encoding the dataframe 
    df_new = df_new.apply(lambda x: object_to_int(x))
    #inputs and target selection 
    X = df_new.drop(columns = ['Churn'])
    y = df_new['Churn'].values
    #train test split (Allowing the user to choose the optimal train/test split percentage)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state = random_state, stratify=y)
    #Standardizing the variables 
    df_std = pd.DataFrame(StandardScaler().fit_transform(df_new[num_cols].astype('float64')),columns=num_cols)
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    if option == 'KNN': 
        knn_model = KNeighborsClassifier(n_neighbors = 11) 
        knn_model.fit(X_train,y_train)
        predicted_y = knn_model.predict(X_test)
        return accuracy_score(predicted_y,y_test), classification_report(y_test, predicted_y),df_new,df
    elif option == 'SVC': 
        svc_model = SVC(random_state = 1)
        svc_model.fit(X_train,y_train)
        predicted_y = svc_model.predict(X_test)
        return accuracy_score(predicted_y,y_test), classification_report(y_test,predicted_y),df_new,df
    elif option == 'RF': 
        model_rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
        model_rf.fit(X_train, y_train)
        predicted_y = model_rf.predict(X_test)
        return accuracy_score(y_test, predicted_y), classification_report(y_test,predicted_y),df_new,df
    elif option == 'LR': 
        lr_model = LogisticRegression()
        lr_model.fit(X_train,y_train)
        predicted_y = lr_model.predict(X_test)
        return accuracy_score(predicted_y,y_test), classification_report(y_test,predicted_y),df_new,df
    elif option == 'DT': 
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train,y_train)
        predicted_y = dt_model.predict(X_test)
        return accuracy_score(predicted_y,y_test), classification_report(y_test,predicted_y),df_new,df
    elif option == 'Adaboost': 
        a_model = AdaBoostClassifier()
        a_model.fit(X_train,y_train)
        predicted_y = a_model.predict(X_test)
        return accuracy_score(predicted_y,y_test), classification_report(y_test,predicted_y),df_new,df
    elif option == 'Gradient Boosting': 
        gb = GradientBoostingClassifier()
        gb.fit(X_train, y_train)
        predicted_y = gb.predict(X_test)
        return accuracy_score(predicted_y,y_test), classification_report(y_test,predicted_y),df_new,df
    elif option == 'Voting Classifier': 
        clf1 = GradientBoostingClassifier()
        clf2 = LogisticRegression()
        clf3 = AdaBoostClassifier()
        eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
        eclf1.fit(X_train, y_train)
        predicted_y = eclf1.predict(X_test)
        return accuracy_score(predicted_y,y_test), classification_report(y_test,predicted_y),df_new,df


def visualize(df): 
    g_labels = ['Male', 'Female']
    c_labels = ['No', 'Yes']
    # Create subplots: use 'domain' type for Pie subplot
    fig1 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig1.add_trace(go.Pie(labels=g_labels, values=df['gender'].value_counts(), name="Gender"),
                  1, 1)
    fig1.add_trace(go.Pie(labels=c_labels, values=df['Churn'].value_counts(), name="Churn"),
                  1, 2)
    # Use `hole` to create a donut-like pie chart
    fig1.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)
    fig1.update_layout(
        title_text="Gender and Churn Distributions",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                     dict(text='Churn', x=0.84, y=0.5, font_size=20, showarrow=False)])
    fig2 = px.histogram(df, x="Churn", color="Contract", barmode="group", title="<b>Customer contract distribution<b>")
    fig2.update_layout(width=700, height=500, bargap=0.1)
    labels = df['PaymentMethod'].unique()
    values = df['PaymentMethod'].value_counts()
    fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig3.update_layout(title_text="<b>Payment Method Distribution</b>")
    fig4 = px.histogram(df, x="Churn", color="PaymentMethod", title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
    fig4.update_layout(width=700, height=500, bargap=0.1)
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
      x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
           ["Female", "Male", "Female", "Male"]],
      y = [965, 992, 219, 240],
      name = 'DSL',
    ))
    fig5.add_trace(go.Bar(
      x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
           ["Female", "Male", "Female", "Male"]],
      y = [889, 910, 664, 633],
      name = 'Fiber optic',
    ))
    fig5.add_trace(go.Bar(
      x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
           ["Female", "Male", "Female", "Male"]],
      y = [690, 717, 56, 57],
      name = 'No Internet',
    ))
    fig5.update_layout(title_text="<b>Churn Distribution w.r.t. Internet Service and Gender</b>")
    color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
    fig6 = px.histogram(df, x="Churn", color="Dependents", barmode="group", title="<b>Dependents distribution</b>", color_discrete_map=color_map)
    fig6.update_layout(width=700, height=500, bargap=0.1)
    color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
    fig7 = px.histogram(df, x="Churn", color="Partner", barmode="group", title="<b>Churn distribution w.r.t. Partners</b>", color_discrete_map=color_map)
    fig7.update_layout(width=700, height=500, bargap=0.1)
    color_map = {"Yes": '#00CC96', "No": '#B6E880'}
    fig8 = px.histogram(df, x="Churn", color="SeniorCitizen", title="<b>Churn distribution w.r.t. Senior Citizen</b>", color_discrete_map=color_map)
    fig8.update_layout(width=700, height=500, bargap=0.1)
    color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
    fig9 = px.histogram(df, x="Churn", color="OnlineSecurity", barmode="group", title="<b>Churn distribution w.r.t Online Security</b>", color_discrete_map=color_map)
    fig9.update_layout(width=700, height=500, bargap=0.1)
    color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
    fig10 = px.histogram(df, x="Churn", color="PaperlessBilling",  title="<b>Churn distribution w.r.t. Paperless Billing</b>", color_discrete_map=color_map)
    fig10.update_layout(width=700, height=500, bargap=0.1)
    fig11 = px.histogram(df, x="Churn", color="TechSupport",barmode="group",  title="<b>Churn distribution w.r.t. Tech Support</b>")
    fig11.update_layout(width=700, height=500, bargap=0.1)
    color_map = {"Yes": '#00CC96', "No": '#B6E880'}
    fig12 = px.histogram(df, x="Churn", color="PhoneService", title="<b>Churn Distribution w.r.t. Phone Service</b>", color_discrete_map=color_map)
    fig12.update_layout(width=700, height=500, bargap=0.1)
    fig13 = px.box(df, x='Churn', y = 'tenure')
    fig13.update_yaxes(title_text='Tenure (Months)', row=1, col=1)
    fig13.update_xaxes(title_text='Churn', row=1, col=1)
    fig13.update_layout(autosize=True, width=750, height=600,
        title_font=dict(size=25, family='Courier'),
        title='<b>Tenure vs Churn</b>',
    )
    return fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11,fig12,fig13  

def take_input(filepath): 
    df = pd.read_csv(filepath)
    processed_df = preprocess(df)
    fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11,fig12,fig13 = visualize(processed_df)
    return fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11,fig12,fig13, processed_df