import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import warnings

# --- Setup: Download NLTK data (runs only once) ---
@st.cache_resource
def setup_nltk_data():
    """Downloads required NLTK data packages if not already present."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

# Run the setup function at the start of the app
setup_nltk_data()

# --- Model Training (Cached for efficiency) ---
@st.cache_resource
def load_and_train_model():
    """
    Loads the news dataset, preprocesses the text, trains a
    PassiveAggressiveClassifier model, and returns a sample of real news.
    """
    # Load the dataset
    df = pd.read_csv('news.csv')

    # Drop rows with missing values for cleaner data
    df.dropna(inplace=True)
    
    # Get a sample of 3 real news articles to display as examples
    real_news_sample = df[df['label'] == 'REAL'].sample(3, random_state=101)

    # Combine title and text for a richer feature set
    df['full_text'] = df['title'] + ' ' + df['text']

    # Define features (X) and labels (y)
    X = df['full_text']
    y = df['label']

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vectorized = vectorizer.fit_transform(X)

    # Initialize and train the PassiveAggressiveClassifier
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_vectorized, y)

    return vectorizer, model, real_news_sample

# Load the trained model, vectorizer, and news samples
with st.spinner('Training model... This may take a moment.'):
    vectorizer, model, real_news_sample = load_and_train_model()

# --- Streamlit User Interface ---
st.title("📰 Fake News Detector")
st.write(
    "Enter the text from a news article below to determine if it is likely "
    "to be REAL or FAKE news."
)

# Create a text area for user input without a placeholder
news_text = st.text_area(
    "Enter News Article Text:",
    height=250,
)

# Create a button to trigger the analysis
if st.button("Analyze News"):
    if news_text:
        # 1. Vectorize the user's input text
        vectorized_text = vectorizer.transform([news_text])

        # 2. Predict using the trained model
        prediction = model.predict(vectorized_text)

        # 3. Display the result
        st.subheader("Analysis Result")
        if prediction[0] == 'FAKE':
            st.error("This article appears to be FAKE news. 🚨")
        else:
            st.success("This article appears to be REAL news. ✅")
    else:
        st.warning("Please enter some text from a news article to analyze.")

st.markdown(
    "--- \n *Powered by a PassiveAggressiveClassifier trained on a public news dataset.*"
)

# --- Display News Examples ---
st.subheader("💡 Examples of Real News from the Dataset")
st.write("You can test the detector with articles like these:")

for index, row in real_news_sample.iterrows():
    with st.expander(f"**{row['title']}**"):
        st.write(row['text'])

