import pandas as pd
import re
import nltk
import pickle
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

with open ('TF.pkl','rb') as TF_pkl :
    TF=pickle.load(TF_pkl)
training_data_csv=pd.read_csv("Data.csv")
training_data_csv
# tweets = train_data[["Messages"]]
# subtask_a_labels = train_data[["subtask_a"]]
messages=training_data_csv["Messages"].tolist()
# messages_type=training_data_csv["Offensive Type"].tolist()
messages_type_list=training_data_csv["Offensive_Type"].values.tolist()
# print(messages)
messages_copy=[]
for k  in messages:
  messages_copy.append(k)
with open ('Classifier_RF.pkl','rb') as file_classifier_pkl:
    Classifier_text=pickle.load(file_classifier_pkl)



def preprocess(df):
  messages=df

  # messages_type=df["Offensive Type"].tolist()
  #messages_type_list=df.Offensive_Type.map(dict(YES=1, NOT=0))
  # print(messages)
  print(messages)



  messages_clean=[]
  message_clean_str =re.sub(r'[^a-zA-Z]', ' ', messages[0])
  stop_words=set(stopwords.words('english'))
  print("." in stop_words)
  # re.sub(r'[^a-zA-Z]', ' ', tweet)
  # print(stop_words)
  data_in_line_format_list=[]
          # tweet = tweet.replace(noise, '')
  # reg_allow="[A-Za-z]"
  for  message in messages:
    requires_list=[]

    # for noise in noises:
    #       message = message.replace(noise, '')
    # message.replace(noises[0],'')
    # message.replace(noises[1],'')
    # message.replace(noises[2],'')
    # message.replace(noises[3],'')
    # message.replace(noises[4],'')
    # message.replace(noises[5],'')
    message= message.replace('@USER','')
    message= message.replace('\'ve','')
    message= message.replace('n\'t','')
    message= message.replace('\'s','')
    message= message.replace('URL','')
    message= message.replace('\'m','')
    message=re.sub(r'[^a-zA-Z]', ' ', message)
    messa_list=word_tokenize(message.lower())
    for texxtt in messa_list:
    
      if (texxtt.lower()  not in  stop_words ):
        texot= wordnet_lemmatizer.lemmatize(texxtt.lower())
        texot = lancaster_stemmer.stem(texot)
        if (len(texot)>1):
          requires_list.append(texot.lower())
          # data_in_line_format= " ".join+(requires_list[-1])
    messages_clean.append(requires_list)
    data_in_line_format_list.append( " ".join(requires_list))
  # messages_clean[:7]
  TF_IDF=TF.transform(data_in_line_format_list).toarray()
  return TF_IDF

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import numpy

# with open ('bigram_count.pkl','rb') as file_bigram_count_pkl:
#   bigram_count=pickle.load(file_bigram_count_pkl)
# with open ('matrix.pkl','rb') as file_matrix_pkl:
#   model=pickle.load(file_matrix_pkl)


# print(y_train)
# train_messages, test_messages, train_message_type,test_message_type=train_test_split(TF_IDF,messages_type_list,test_size=0.25)
# print(messages_type_list) 


# train_accuracy_found=accuracy_score(y_train,Classifier_text.predict(X_train))
# # accuracy = accuracy_score(train_labels, classifier.predict(train_vectors))
# print("Training Accuracy:", train_accuracy_found)
# # predict_test = Classifier_text.predict(test_tfidf)
# test_accuracy = accuracy_score(y_test,Classifier_text.predict(X_test))
# print("Test Accuracy:", test_accuracy)
# print("Confusion Matrix:", )
# # print(confusion_matrix(test_message_type, predict_test))


# classifier = SVC()
#         classifier = GridSearchCV(classifier, {'C':[0.001, 0.01, 0.1, 1, 10]}, cv=3, n_jobs=4)
#         classifier.fit(train_vectors, train_labels)
         #classifier = classifier.best_estimator_from sklearn.metrics import accuracy_score, confusion_matrix

def message_detection(input__):
  #print(text)
  #text_list=text.split()

  #ans={}
  a=[input__]


  train=preprocess(a)
  print(train)
  # print(accuracy_score(test,Classifier_text.predict(train)))
  result=Classifier_text.predict(train)
  if (result[0]=="OFF"):
    result_to_return= "Offensive"
  else:
    result_to_return= "Not Offensive"

  #ans[a[0]]=result_to_return
  return result_to_return
# a="you you can be a good person but sometime  you are motherfucker"
# print(message_detection(a))
