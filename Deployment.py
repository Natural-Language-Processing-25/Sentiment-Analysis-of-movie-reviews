import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''
To run deployment, open enviroment terminal then:
1. cd to project folder partition. ex: cd d:
2. cd to project folder. ex: cd "\Sentiment-Analysis-of-movie-reviews"
3. run this command: streamlit run deployment.py
'''

######################## Functions & Definitions ########################

def classify_text_features(vectorizer, text_data, model):
    X = vectorizer.transform(text_data)
    y_pred = model.predict(X)
    return y_pred

# Display animation
def load_lottie(url): 
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Loading the vectorizer
vectorizer = joblib.load(r"Models\Vectorizer.joblib")

# Loading the models 
DT_model = joblib.load(r"Models\DT.joblib")
LR_L1_model = joblib.load(r"Models\LR_L1.joblib")
LR_L2_model = joblib.load(r"Models\LR_L2.joblib")
NB_model = joblib.load(r"Models\\NB.joblib")
RF_model = joblib.load(r"Models\RF.joblib")
SVM_Linear_model = joblib.load(r"Models\SVM_Linear.joblib")
SVM_Poly_model = joblib.load(r"Models\SVM_Poly.joblib")
SVM_rbf_model = joblib.load(r"Models\SVM_rbf.joblib")

# Models dictionary
models = {
    "Decision Tree": DT_model,
    "Logistic Regression: L1": LR_L1_model,
    "Logistic Regression: L2": LR_L2_model,
    "Naive Bayes": NB_model,
    "Random Forest": RF_model,
    "SVM: Linear": SVM_Linear_model,
    "SVM: Poly": SVM_Poly_model,
    "SVM: RBF": SVM_rbf_model,
}

# Confusion matrices dictionary
confusionMatrices = {
    "Decision Tree": r"",
    "Logistic Regression: L1": r"",
    "Logistic Regression: L2": r"",
    "Naive Bayes": r"",
    "Random Forest": r"",
    "SVM: Linear": r"",
    "SVM: Poly": r"",
    "SVM: RBF": r"",
}

# Accuracies
accuracies = {
    "Decision Tree": r"Accuracies\DT.txt",
    "Logistic Regression: L1": r"Accuracies\LR_L1.txt",
    "Logistic Regression: L2": r"Accuracies\LR_L2.txt",
    "Naive Bayes": r"Accuracies\NB.txt",
    "Random Forest": r"Accuracies\RF.txt",
    "SVM: Linear": r"Accuracies\SVM_Linear.txt",
    "SVM: Poly": r"Accuracies\SVM_Poly.txt",
    "SVM: RBF": r"Accuracies\SVM_rbf.txt",
}

column = ['text']

######################## Main Display ########################

#Set page configuration
st.set_page_config(
    page_title = 'Sentiment Analysis of Movie Reviews',
    page_icon = '::magnifier::',
    initial_sidebar_state = 'expanded',
)

# Center align the header
st.markdown("<h1 style='text-align: center;'>Sentiment Analysis of Movie Reviews System</h1>", unsafe_allow_html=True)

# Display animation at the top
gif_path = r"Visualization\GIF.gif"

# Create three columns and put the image in the center one
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(gif_path)

# Sidebar Design
with st.sidebar:
    choose = option_menu(None, ["About", "Predictions", "Graphs"],
                        icons = [ 'house','kanban', 'book'],
                        menu_icon = "app-indicator", default_index = 0,
                        styles ={ 
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#428DFF"},
    }
    )

# About Page
if choose == 'About':
    
    st.markdown("<h3 style='text-align: center;'>Sentiment Analysis of Movie Reviews System About:</h3>", unsafe_allow_html=True)
    st.write('---')
    st.write("##### ")
    
# Predictions Page
elif choose == 'Predictions':
    st.markdown("<h3 style='text-align: center;'>Sentiment Analysis of Movie Reviews Predictions:</h3>", unsafe_allow_html=True)
    st.write('---')

    # Select model using dropdown menu
    selected_model = st.selectbox("Select Model", list(models.keys()), key = 'model_selection')
    
    # Open the file in read mode
    file_path = accuracies[selected_model]  
    with open(file_path, "r") as file:
        file_contents = file.read()
        session_state = st.session_state
        
    if 'Show_matrix_and_accuracy' not in session_state:
        session_state.Show_matrix_and_accuracy = False
    
    # Toggles button so that when it's pressed, it stays pressed and doesn't refresh.
    if st.button('Show matrix and accuracy'):
        session_state.Show_matrix_and_accuracy = True
        
    if session_state.Show_matrix_and_accuracy:
        # Display accuracy and confussion matrix using the selected model
        st.write("##### This is the accuracy of the", selected_model , " : ", file_contents, '%')
        st.write(" ")
        st.write("##### This is the confussion matrix for the ", selected_model ,  " :")
        st.write(" ")
        st.image(confusionMatrices[selected_model])
    st.write('---')

    text_input = st.text_area("Enter text for analysis:", height = 150)

    if st.button('Classify'):
        if text_input:
            classification = classify_text_features(vectorizer, [text_input], models[selected_model])
            st.write(f"### Predicted Result ({selected_model}):")
            if classification[0] == 1:  # Assuming 1 represents positive
                st.markdown("<h2 style='color: green;'>Positive Review 👍🏼</h2>", unsafe_allow_html=True)
                st.balloons()
            elif classification[0] == 0:  # Assuming 0 represents negative
                st.markdown("<h2 style='color: red;'>Negative Review 👎🏼</h2>", unsafe_allow_html=True)
            else:
                st.write("Neutral or Undetermined")
        else:
            st.warning("Please enter some text to analyze.")
                
elif choose == 'Graphs':
    st.markdown("<h3 style='text-align: center;'>Sentiment Analysis of Movie Reviews System Graphs :</h3>", unsafe_allow_html=True)
    st.write('---')
    
    # st.write("### ")
    # st.image("")