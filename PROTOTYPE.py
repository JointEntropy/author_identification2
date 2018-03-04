# coding: utf8
import sys


import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
	sample_path = sys.argv[1]
	with open(sample_path,'r') as f:
		data = f.read()
	# Загружаем модели
	with open('logtest90.pkl','rb') as f:
	    classifier = pkl.load(f)
	with open('labelencoder.pkl','rb') as f:
	    le = pkl.load(f)
	with open('tfidf.pkl','rb') as f:
	    vectorizer = pkl.load(f)
	print('Models loaded')
	ALLOWED_CHARS = {chr(chr_idx) for chr_idx in range(ord('а'), ord('я')+1)}
	ALLOWED_CHARS |= set('ё,.—?!: \t\n') 

	data =  data.strip().lower()
	data = ''.join(filter(lambda ch: ch in ALLOWED_CHARS, data))
	data = 	vectorizer.transform([data])
	prediction = classifier.predict(data)
	print(le.inverse_transform(prediction))