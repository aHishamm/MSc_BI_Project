import pandas as pd 
import numpy as np 
from plotly.subplots import make_subplots 
import plotly.graph_objects as go 
import matplotlib.pyplot as plt 
import plotly.express as px 
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
def visualize(df,option): 
    if option == 'Gender and Churn Distribution': 
        g_labels = ['Male', 'Female']
        c_labels = ['No', 'Yes']
        # Create subplots: use 'domain' type for Pie subplot
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace(go.Pie(labels=g_labels, values=df['gender'].value_counts(), name="Gender"),
                      1, 1)
        fig.add_trace(go.Pie(labels=c_labels, values=df['Churn'].value_counts(), name="Churn"),
                      1, 2)
        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)
        fig.update_layout(
            title_text="Gender and Churn Distributions",
            # Add annotations in the center of the donut pies.
            annotations=[dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                         dict(text='Churn', x=0.84, y=0.5, font_size=20, showarrow=False)])
        return fig 
    elif option == 'Customer Contract Distribution': 
        fig = px.histogram(df, x="Churn", color="Contract", barmode="group", title="<b>Customer contract distribution<b>")
        fig.update_layout(width=700, height=500, bargap=0.1)
        return fig 
    elif option == 'Payment Method Distribution': 
        labels = df['PaymentMethod'].unique()
        values = df['PaymentMethod'].value_counts()

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title_text="<b>Payment Method Distribution</b>")
        return fig 
    elif option == 'Payment Method Distribution Churn': 
        fig = px.histogram(df, x="Churn", color="PaymentMethod", title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
        fig.update_layout(width=700, height=500, bargap=0.1)
        return fig
    elif option == 'Churn Distribution w.r.t Internet Service and Gender': 
        fig = go.Figure()
        fig.add_trace(go.Bar(
          x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
               ["Female", "Male", "Female", "Male"]],
          y = [965, 992, 219, 240],
          name = 'DSL',
        ))
        fig.add_trace(go.Bar(
          x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
               ["Female", "Male", "Female", "Male"]],
          y = [889, 910, 664, 633],
          name = 'Fiber optic',
        ))
        fig.add_trace(go.Bar(
          x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
               ["Female", "Male", "Female", "Male"]],
          y = [690, 717, 56, 57],
          name = 'No Internet',
        ))
        fig.update_layout(title_text="<b>Churn Distribution w.r.t. Internet Service and Gender</b>")
        return fig 
    elif option == 'Dependents Distribution Churn':
        color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
        fig = px.histogram(df, x="Churn", color="Dependents", barmode="group", title="<b>Dependents distribution</b>", color_discrete_map=color_map)
        fig.update_layout(width=700, height=500, bargap=0.1)
        return fig 
    elif option == 'Churn Distribution w.r.t Partners': 
        color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
        fig = px.histogram(df, x="Churn", color="Partner", barmode="group", title="<b>Churn distribution w.r.t. Partners</b>", color_discrete_map=color_map)
        fig.update_layout(width=700, height=500, bargap=0.1)
        return fig
    elif option == 'Churn Distribution w.r.t Senior Citizens': 
        color_map = {"Yes": '#00CC96', "No": '#B6E880'}
        fig = px.histogram(df, x="Churn", color="SeniorCitizen", title="<b>Churn distribution w.r.t. Senior Citizen</b>", color_discrete_map=color_map)
        fig.update_layout(width=700, height=500, bargap=0.1)
        return fig 
    elif option == 'Churn Distribution w.r.t Online Security': 
        color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
        fig = px.histogram(df, x="Churn", color="OnlineSecurity", barmode="group", title="<b>Churn distribution w.r.t Online Security</b>", color_discrete_map=color_map)
        fig.update_layout(width=700, height=500, bargap=0.1)
        return fig 
    elif option == 'Churn Distribution w.r.t Paperless Billing': 
        color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
        fig = px.histogram(df, x="Churn", color="PaperlessBilling",  title="<b>Churn distribution w.r.t. Paperless Billing</b>", color_discrete_map=color_map)
        fig.update_layout(width=700, height=500, bargap=0.1)
        return fig 
    elif option == 'Churn Distribution w.r.t Tech Support': 
        fig = px.histogram(df, x="Churn", color="TechSupport",barmode="group",  title="<b>Churn distribution w.r.t. Tech Support</b>")
        fig.update_layout(width=700, height=500, bargap=0.1)
        return fig 
    elif option == 'Churn Distribution w.r.t Phone Service': 
        color_map = {"Yes": '#00CC96', "No": '#B6E880'}
        fig = px.histogram(df, x="Churn", color="PhoneService", title="<b>Churn Distribution w.r.t. Phone Service</b>", color_discrete_map=color_map)
        fig.update_layout(width=700, height=500, bargap=0.1)
        return fig 
    elif option == 'Tenure vs. Churn': 
        fig = px.box(df, x='Churn', y = 'tenure')
        fig.update_yaxes(title_text='Tenure (Months)', row=1, col=1)
        fig.update_xaxes(title_text='Churn', row=1, col=1)
        fig.update_layout(autosize=True, width=750, height=600,
            title_font=dict(size=25, family='Courier'),
            title='<b>Tenure vs Churn</b>',
        )
        return fig  


def take_input(filepath,option): 
    df = pd.read_csv(filepath)
    processed_df = preprocess(df)
    figure = visualize(processed_df,option)
    return figure 




