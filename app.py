import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')


ps = PorterStemmer()

tfidf = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


def transform_text(text):
    """Function for applying all the data preprocessing mentioned above"""
    # converting to lowercase
    text = text.lower()
    # breaking the sentences into words
    text = nltk.word_tokenize(text)
    # removing all the special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.markdown(f"")
        st.header("Spam Message!!")
    else:
        st.header("Not a Spam message")
