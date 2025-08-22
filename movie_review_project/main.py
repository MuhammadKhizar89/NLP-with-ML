import re
import string
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

stop_words=set(stopwords.words('english'))
def remove_stopwords(text):
    word_tokens=word_tokenize(text)
    filtered_sentence=[w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)


def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])

def remove_emojis(text):
    return ''.join([char for char in text if char.isascii()])

def predict_review(review):
    pipeline = joblib.load("sentiment_pipeline.pkl")
    review = remove_html_tags(review)
    review = remove_punctuation(review)
    review = remove_emojis(review)
    review = remove_numbers(review)
    review = review.lower()
    review = remove_stopwords(review)
    result=pipeline.predict([review])
    return "Positive" if result == 1 else "Negative"

print(predict_review("I hate this movie"))