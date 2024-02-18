import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from imblearn.over_sampling import SMOTE
from colorama import Fore, Style
import time
from rich.console import Console
from rich.progress import Progress

data = pd.read_csv('creditcard.csv')    
pd.options.display.max_columns = None

sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))

data = data.drop(['Time'], axis = 1)

fraud = data[data['Class'] == 1]
legit = data[data['Class'] == 0]

normal_sample = legit.sample(n=473)
new_data = pd.concat([normal_sample,fraud])

X = new_data.drop(['Class'],axis = 1)
y = new_data['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)



print(accuracy_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(f1_score(y_test,y_pred))

st.title('Credit Card Fraud Detection Model')
input_df = st.text_input('Enter transaction details')
input_df_splited = input_df.split(',')

submit = st.button('Submit')

def success_animation():
    # Add a success icon using HTML and CSS
    success_icon_html = """
    <div style="text-align:center;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="100" height="100" fill="#4CAF50">
            <path d="M0 0h24v24H0z" fill="none"/>
            <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/>
        </svg>
    </div>
    """
    
    # Display the success icon
    st.markdown(success_icon_html, unsafe_allow_html=True)
    st.success("Transaction is Legitimate!")

def error_animation():
    # Error icon
    error_icon_html = """
    <div style="text-align:center;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="100" height="100" fill="#FF5733">
            <path d="M0 0h24v24H0z" fill="none"/>
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
    </div>
    """

    # Display the error icon and error message
    st.markdown(error_icon_html, unsafe_allow_html=True)
    st.error("This appears to be a Fraudulent Transaction!")

if submit:
    features = np.asarray(input_df_splited, dtype = np.float64)
    prediction = rf.predict(features.reshape(1,-1))

    if prediction[0] == 0:
        success_animation()
    elif prediction[0] == 1:
        error_animation()
    else:
        st.write('Invalid Entry')
