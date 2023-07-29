from django.shortcuts import render, render
import pickle
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
#loading model
file = open("pickle/model.pkl", 'rb')
RF = pickle.load(file)
file.close()

#loading trained vectorizer
file = open('pickle/vectorizer.pkl', 'rb')
vectorizer = pickle.load(file)
file.close()


def cleaning_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_label(n):
    if n == 0:
        return 'Fake News'
    if n == 1:
        return 'Real news'


def manual_testing(news):
    testing_news = {'text': [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(cleaning_text)
    new_x_test = new_def_test['text']
    new_xv_test = vectorizer.transform(new_x_test)
    pred_RF = RF.predict(new_xv_test)
    return output_label(pred_RF[0])


def home(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        #TODO preprocess the text and model predict
        result = manual_testing(text)
        context = {'result': result}
        return render(request, 'base/home.html', context)

    return render(request, 'base/home.html')