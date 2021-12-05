# -*- coding: utf-8 -*-
"""2018268_NLP_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14MPzUVRFcQ9oW6QG8buqFVPgmheHrmIR

**Importing Libraries**.
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.naive_bayes import GaussianNB, MultinomialNB
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import numpy

"""**Importing Files From Google Drive**"""

!gdown --id 12VfzmOJ40IRf_GJYD7G9lFKU8cEJtTXP
# https://drive.google.com/file/d/12VfzmOJ40IRf_GJYD7G9lFKU8cEJtTXP
!gdown --id 1orcHrValg3MsP5f3hjn1dCBg4AVU33Ct
!gdown --id 1cClNTOpY_-4CJ3SWLKsGqQlKU4xi7ayX1d2UHV7uVKY


!gdown --id 1FUIQLXPCkBSDwQ7_fWjYQPaVeyEIFkuC
# https://drive.google.com/file/d/1FUIQLXPCkBSDwQ7_fWjYQPaVeyEIFkuC/

!gdown --id 1-GBb4LBojCgjSdOS8zr4RqdyBuAQpt0n

training_data_csv=pd.read_csv("Data.csv")
training_data_csv

def preprocess(df):
  messages=df["Messages"].tolist()
  # messages_type=df["Offensive Type"].tolist()
  messages_type_list=df.Offensive_Type.map(dict(YES=1, NOT=0))
  # print(messages)



  messages_clean=[]
  message_clean_str =re.sub(r'[^a-zA-Z]', ' ', messages[0])
  stop_words=set(stopwords.words('english'))
  print("." in stop_words)

  data_in_line_format_list=[]

  for  message in messages:
    requires_list=[]

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
  TF_IDF=TfidfVectorizer().fit(data_in_line_format_list).transform(data_in_line_format_list).toarray()
  return TF_IDF,messages_type_list

TF_IDF, messages_type_list=preprocess(training_data_csv)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression



# X_train,X_test,y_train,y_test=train_test_split(TF_IDF,messages_type_list,test_size=0.25)
train_messages, test_messages, train_message_type,test_message_type=train_test_split(TF_IDF,messages_type_list,test_size=0.25)



"""MNB Classifier"""

Classifier_MNB = MultinomialNB(alpha=0.7)
Classifier_MNB.fit(train_messages,train_message_type)

accuracy = accuracy_score(train_message_type, Classifier_MNB.predict(train_messages))
print("Train accuracy:---")
print( accuracy)
accuracy = accuracy_score(test_message_type, Classifier_MNB.predict(test_messages))
print("Test accuracy:---")
print( accuracy)

# # X_train,X_test,y_train,y_test=train_test_split(TF_IDF,messages_type_list,test_size=0.25)

# # train_messages, test_messages, train_message_type,test_message_type=train_test_split(TF_IDF,messages_type_list,test_size=0.25)

# train_accuracy_found=accuracy_score(train_message_type,Classifier_text.predict(train_messages))
# # accuracy = accuracy_score(train_labels, classifier.predict(train_vectors))
# print("Training Accuracy:", train_accuracy_found)

# test_accuracy = accuracy_score(test_message_type,Classifier_text.predict(test_messages))

# print(test_accuracy,train_accuracy_found)

# # classifier = SVC()
# #         classifier = GridSearchCV(classifier, {'C':[0.001, 0.01, 0.1, 1, 10]}, cv=3, n_jobs=4)
# #         classifier.fit(train_vectors, train_labels)
# #         classifier = classifier.best_estimator_from sklearn.metrics import accuracy_score, confusion_matrix

#Random Forest

"""Random Forest Classifier"""

# Classifier_RF=classify(TF_IDF, messages_type_list) # {MNB, KNN, SVM, DT, RF, LR}
# import pickle
Classifier_RF = RandomForestClassifier(max_depth=800, min_samples_split=5)
parameters = {'n_estimators': [50, 100, 150], 'criterion':['gini','entropy'], }
Classifier_RF = GridSearchCV(Classifier_RF, parameters, cv=3, n_jobs=4).fit(train_messages, train_message_type)
Classifier_RF = Classifier_RF.best_estimator_
with open ('Classifier_RF.pkl','wb') as Classsifier_RF_pkl:
	pickle.dump(Classifier_RF,Classsifier_RF_pkl)

accuracy = accuracy_score(train_message_type, Classifier_RF.predict(train_messages))
print("Train accuracy:---")
print( accuracy)
accuracy = accuracy_score(test_message_type, Classifier_RF.predict(test_messages))
print("Test accuracy:---")
print( accuracy)



"""LR Classifier """

Classifier_LR = LogisticRegression(multi_class='auto', solver='newton-cg',)
Classifier_LR = GridSearchCV(Classifier_LR, {"C":np.logspace(-3,3,7), "penalty":["l2"]}, cv=3, n_jobs=4).fit(train_messages, train_message_type)
Classifier_LR = Classifier_LR.best_estimator_

with open ('Classifier_LR.pkl','wb') as Classsifier_LR_pkl:
	pickle.dump(Classifier_LR,Classsifier_LR_pkl)

accuracy = accuracy_score(train_message_type, Classifier_LR.predict(train_messages))
print("Train accuracy:---")
print( accuracy)
accuracy = accuracy_score(test_message_type, Classifier_LR.predict(test_messages))
print("Test accuracy:---")
print( accuracy)



# accuracy_score(test,Classifier_text.predict(train))

"""DT Classifier"""

Classifier_DT = DecisionTreeClassifier(max_depth=800, min_samples_split=5)
parameters = {'criterion':['gini','entropy']}
Classifier_DT = GridSearchCV(Classifier_DT, parameters, cv=3, n_jobs=4).fit(train_messages, train_message_type)
Classifier = classifier.best_estimator_
with open ('Classifier_DT.pkl','wb') as Classsifier_DT_pkl:
	pickle.dump(Classifier_DT,Classsifier_DT_pkl)

accuracy = accuracy_score(train_message_type, Classifier_DT.predict(train_messages))
print("Train accuracy:---")
print( accuracy)
accuracy = accuracy_score(test_message_type, Classifier_DT.predict(test_messages))
print("Test accuracy:---")
print( accuracy)



"""SVM Classifier

"""

Classifier_SVM = SVC()
Classifier_SVM = GridSearchCV(Classifier_SVM, {'C':[0.001, 0.01, 0.1, 1, 10]}, cv=3, n_jobs=4).fit(train_messages, train_message_type)
Classifier_SVM = Classifier_SVM.best_estimator_
with open ('Classifier_DT.pkl','wb') as Classsifier_SVM_pkl:
	pickle.dump(Classifier_DT,Classsifier_SVM_pkl)

accuracy = accuracy_score(train_message_type, Classifier_SVM.predict(train_messages))
print("Train accuracy:---")
print( accuracy)
accuracy = accuracy_score(test_message_type, Classifier_SVM.predict(test_messages))
print("Test accuracy:---")
print( accuracy)

"""KNN Classifier"""

Classfier_KNN = KNeighborsClassifier(n_jobs=4)
papameters = {'n_neighbors': [3,5,7,9], 'weights':['uniform', 'distance']}
Classfier_KNN = GridSearchCV(Classfier_KNN, papameters, cv=3, n_jobs=4).fit(train_messages, train_message_type)
Classfier_KNN = Classfier_KNN.best_estimator_
with open ('Classifier_KNN.pkl','wb') as Classsifier_KNN_pkl:
	pickle.dump(Classifier_KNN,Classsifier_KNN_pkl)



accuracy = accuracy_score(train_message_type, Classifier_KNN.predict(train_messages))
print("Train accuracy:---")
print( accuracy)
accuracy = accuracy_score(test_message_type, Classifier_KNN.predict(test_messages))
print("Test accuracy:---")
print( accuracy)
