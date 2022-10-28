import pandas as pd
import streamlit as slt
from prediction_file import predict_cnn

slt.set_page_config(
     page_title='Churn Modelling'
)

slt.title("Customer Retention Deep learning model.")

slt.header("Teaching systems to predict, If Customer will stay with the bank or leave.")

credit_score = slt.slider("Credit Score",min_value=350,max_value=850)

geography = slt.selectbox(
    'Geography',
    ("France","Germany","Spain"))
gender = slt.radio("Gender",("Male","Female"))

Age = slt.number_input("Age",min_value=18,max_value=70)

Tenure = slt.number_input("Tenure",min_value=0,max_value=10)
Balance = slt.slider("Bank balance",min_value=0,max_value=500000)
NumOfProducts = slt.select_slider("Number of Products",[1,2,3,4])
HasCrCard = slt.radio("Has credit card",("Yes","No"))
IsActiveMember = slt.radio("Is active member",("Yes","No"))
EstimatedSalary = slt.slider("Estimated salary",min_value=0,max_value=500000)
result = slt.button("Predict")
Germany = 0
Spain = 0
slt.sidebar.subheader("Accuracy and loss of a model")
slt.sidebar.line_chart(pd.read_csv("F:\Project\churn_model\model_history.csv")[["acc","val_acc"]])
slt.sidebar.line_chart(pd.read_csv("F:\Project\churn_model\model_history.csv")[["loss","val_loss"]])
if result:
     if geography == "France":
         Germany = 0
         Spain = 0
     elif geography == "Germany":
         Germany = 1
         Spain = 0
     elif geography == "Spain":
         Germany = 0
         Spain = 1

     if HasCrCard == "Yes":
         HasCrCard = 1
     elif HasCrCard == "No":
         HasCrCard = 0

     if IsActiveMember == "Yes":
         IsActiveMember = 1
     elif IsActiveMember == "No":
         IsActiveMember = 0

     if gender == "Male":
         Male = 1
     elif gender == "Female":
         Male = 0

     slt.write("Your result will be soon displayed. Predicting..")
     slt.spinner()
     with slt.spinner(text="Work in Progress..."):
         data = pd.DataFrame({"CreditScore":credit_score,"Age":Age,"Tenure":Tenure,"Balance":Balance,"NoOfProducts":NumOfProducts,"HasCrCard":HasCrCard,"IsActiveMember":IsActiveMember,"EstimatedSalary":EstimatedSalary,"Germany":Germany,"Spain":Spain,"Male":Male},index=[0])
         y_pred = predict_cnn(data)
         if y_pred == 1:
             slt.write("The Customer would be leaving the bank so please maintain better relationship with the customer....")
         else:
             slt.write("The customer will be continuing the bank services...")

