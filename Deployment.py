import streamlit as st
import joblib
import requests
from streamlit_option_menu import option_menu
from nltk.stem import PorterStemmer
import spacy
nlp = spacy.load("en_core_web_sm")

#Set page configuration
st.set_page_config(
    page_title = 'Sentiment Analysis of Movie Reviews',
    page_icon = '::star::',
    initial_sidebar_state = 'expanded',
)

######################## Functions & Definitions ########################

def apply_lemmatization(tokens):
    return [token.lemma_ for token in nlp(' '.join(tokens))]

def preprocess_text(text):
    doc = nlp(text_input)
    filtered_tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    lemmatized_tokens = apply_lemmatization(filtered_tokens)
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def classify_text_features(vectorizer, text, model):
    X = vectorizer.transform(text)
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

# Confusion matrices
confusionMatrices = {
    "Decision Tree": r"Visualization\confusion_matrix_Decision_Tree.png",
    "Logistic Regression: L1": r"Visualization\confusion_matrix_Logistic_Regression_L1.png",
    "Logistic Regression: L2": r"Visualization\confusion_matrix_Logistic_Regression_L2.png",
    "Naive Bayes": r"Visualization\confusion_matrix_Naive_Bayes.png",
    "Random Forest": r"Visualization\confusion_matrix_Random_Forest.png",
    "SVM: Linear": r"Visualization\confusion_matrix_SVM_Linear.png",
    "SVM: Poly": r"Visualization\confusion_matrix_SVM_Poly.png",
    "SVM: RBF": r"Visualization\confusion_matrix_SVM_RBF.png",
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

# Center align the header
st.markdown("<h1 style='text-align: center; color: #649FD2;'>🎬 Sentiment Analysis of Movie Reviews System</h1>",unsafe_allow_html=True)

# Sidebar Design
with st.sidebar:
    choose = option_menu(None, ["About", "Predictions", "Graphs"],
                        icons = [ 'info-circle','clipboard-data', 'graph-up-arrow'],
                        menu_icon = "app-indicator", default_index = 0,
                        styles ={ 
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#649FD2"},
    }
    )

# About Page
if choose == 'About':
    # Display animation at the top
    gif_path = r"Visualization\\GIF.gif"

    # Create three columns and put the image in the center one
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(gif_path)
    st.write('---')
    aboutText = """
<p>Welcome to our Sentiment Analysis Pipeline — your smart assistant for understanding the mood behind movie reviews! 🎭💬</p>
<br>
<p>This system is designed to classify text reviews as <b>Positive 👍</b> or <b>Negative 👎</b> by processing a rich dataset of movie reviews stored in text files. It performs:</p>
<ul>
  <li>🔄 <b>Preprocessing</b>: Cleaning and preparing text data</li>
  <li>🔍 <b>Augmentation</b>: Enhancing data diversity for better learning</li>
  <li>⚙️ <b>Feature Extraction</b>: Transforming text into meaningful numerical features</li>
  <li>📊 <b>Visualization</b>: Displaying insightful results using dimensionality reduction techniques</li>
</ul>
<br>
<p>Built with the power of Python in a Jupyter Notebook, this system leverages cutting-edge libraries like:</p>
<ul>
  <li>spaCy for advanced natural language processing 🧠</li>
  <li>NLTK for linguistic data manipulation 📚</li>
  <li>scikit-learn for machine learning magic 🤖</li>
  <li>Matplotlib for beautiful data visualizations 📈</li>
</ul>
"""
    st.markdown(aboutText, unsafe_allow_html=True)

# Predictions Page
elif choose == 'Predictions':
    st.markdown("<h3 style='text-align: center;'>🔮 Sentiment Analysis of Movie Reviews Predictions:</h3>", unsafe_allow_html=True)
    st.write('---')

    models_list = list(models.keys())
    # Track previous selection
    if 'prev_model_index' not in st.session_state:
        st.session_state.prev_model_index = 0
    if 'Show_matrix_and_accuracy' not in st.session_state:
        st.session_state.Show_matrix_and_accuracy = False

    # Select model using dropdown menu
    selected_model = st.selectbox("Select Model", models_list, key='model_selection')
    current_index = models_list.index(selected_model)

    # If selection changed, reset display flag
    if st.session_state.prev_model_index != current_index:
        st.session_state.Show_matrix_and_accuracy = False
    st.session_state.prev_model_index = current_index

    # Button to show matrix and accuracy
    if st.button('Show matrix and accuracy'):
        st.session_state.Show_matrix_and_accuracy = True

    if st.session_state.Show_matrix_and_accuracy:
        file_path = accuracies[selected_model]
        with open(file_path, "r") as file:
            file_contents = file.read()
        st.write("##### This is the accuracy of the", selected_model , " : ", file_contents, '%')
        st.write(" ")
        st.image(confusionMatrices[selected_model])
    st.write('---')

    text_input = st.text_area("Enter text for analysis:", height = 150)
    if st.button('Classify'):
        if text_input:
            processed_text = preprocess_text(text_input)
            st.write('---')
            classification = classify_text_features(vectorizer, [processed_text], models[selected_model])
            st.write(f"### Predicted Result ({selected_model}):")
            if classification[0] == 1: 
                st.markdown("<h2 style='color: green;'>Positive Review 👍🏼</h2>", unsafe_allow_html=True)
                st.balloons()
            elif classification[0] == 0:
                st.markdown("<h2 style='color: red;'>Negative Review 👎🏼</h2>", unsafe_allow_html=True)
            st.write('---')
            st.write("#### Processed text:")
            st.write(processed_text)
        else:
            st.warning("Please enter some text to analyze.")
            
elif choose == 'Graphs':
    st.markdown("<h3 style='text-align: center;'>📊 Sentiment Analysis of Movie Reviews System Graphs :</h3>", unsafe_allow_html=True)
    st.write('---')
    
    st.write("### BERT confussion Matrx Graph:")
    st.image(r"Visualization\bert_confusion_matrix.png")
    st.write('---')
    
    st.write("### Accuracies Comparison Graphs:")
    st.image(r"Visualization\model_accuracy_comparison.png")
    st.write('---')
    
    st.write("### TF-IDF Graphs:")
    st.image(r"Visualization\tfidf_heatmap.png")
    st.image(r"Visualization\tfidf_top_features.png")
    
    st.write("### Word Cloud Graph:")
    st.image(r"Visualization\wordcloud.png")
    st.write('---')
    
    st.write("### PCA Graphs:")
    st.image(r"Visualization\pca_tfidf_LR_L1_predictions.png")
    st.image(r"Visualization\pca_tfidf_LR_L2_predictions.png")
    st.image(r"Visualization\pca_tfidf_Naive_Bayes_predictions.png")
    st.image(r"Visualization\pca_tfidf_Decision_Tree_predictions.png")
    st.image(r"Visualization\pca_tfidf_Random_Forest_predictions.png")
    st.image(r"Visualization\pca_tfidf_SVM_Linear_predictions.png")
    st.image(r"Visualization\pca_tfidf_SVM_Poly_predictions.png")
    st.image(r"Visualization\pca_tfidf_SVM_RBF_predictions.png")
    st.image(r"Visualization\pca_tfidf_true.png")
    st.write('---')
    
    st.write("### ROC Curve Graph:")
    st.image(r"Visualization\roc_curve_all_models.png")
    st.write('---')

    st.write("### TSNE Graphs:")
    st.image(r"Visualization\tsne_tfidf_LR_L1_predictions.png")
    st.image(r"Visualization\tsne_tfidf_LR_L2_predictions.png")
    st.image(r"Visualization\tsne_tfidf_Naive_Bayes_predictions.png")
    st.image(r"Visualization\tsne_tfidf_Decision_Tree_predictions.png")
    st.image(r"Visualization\tsne_tfidf_Random_Forest_predictions.png")
    st.image(r"Visualization\tsne_tfidf_SVM_Linear_predictions.png")
    st.image(r"Visualization\tsne_tfidf_SVM_Poly_predictions.png")
    st.image(r"Visualization\tsne_tfidf_SVM_RBF_predictions.png")
    st.image(r"Visualization\tsne_tfidf_true.png")