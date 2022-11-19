import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Cleaning the messages
def Cleaning(text):
    corpus = []
    wnl = WordNetLemmatizer()

    for sms_string in list(text):
        # Cleaning special character from the sms
        message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_string)
        # Converting the entire sms into lower case
        message = message.lower()
        # to remove numeric digits from string
        message = ''.join([i for i in message if not i.isdigit()])
        # Tokenizing the sms by words
        words = message.split()
        # Removing the stop words
        filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
        # Lemmatizing the words
        lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]
        # Joining the lemmatized words
        message = ' '.join(lemmatized_words)
        # Building a corpus of messages
        corpus.append(message)
    return corpus


tfidf = pickle.load(open('vectorizer500.pkl', 'rb'))
model = pickle.load(open('model500.pkl', 'rb'))

st.title("SMS Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = Cleaning([input_sms])
    # 2. vectorize
    vector_input = tfidf.transform(transformed_sms).toarray()
    # 3. predict
    result = model.predict(vector_input)
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Ham")
