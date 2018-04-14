"""
Ссылки по теме:
- [Работа с текстом в scikit](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

"""
# appendix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

corpus = data.reset_index(drop=True)
X, y = corpus['text'],corpus['author']

# encode classes
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

# do train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# sanity check for stratified split
plt.subplots(1,2,figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(y_train);
plt.subplot(1,2,2)
sns.countplot(y_test);


%%time
from  sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10**5)
vectorizer.fit(X_train)
X_train, X_test = vectorizer.transform(X_train), vectorizer.transform(X_test)


%%time
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
lr = LogisticRegression()
#lr.fit(X_train, y_train)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
lr_params = {
    'C': np.logspace(-1,1, 3),#np.logspace(-4,3, 8),
    'penalty': ['l1','l2']
}
grid = GridSearchCV(lr, lr_params, verbose=2, n_jobs=-1)#, random_state=8)
grid.fit(X_train, y_train)
grid.best_score_, grid.best_params_

from sklearn.metrics import accuracy_score
grid.score(X_test,y_test)



import pickle as pkl
with open('data/logtest90.pkl','wb') as f:
    pkl.dump(grid, f)
with open('data/labelencoder.pkl','wb') as f:
    pkl.dump(le, f)
with open('data/tfidf.pkl','wb') as f:
    pkl.dump(vectorizer, f)

with open('data/logtest90.pkl', 'rb') as f:
    classifier = pkl.load(f)
classifier.score(X_test, y_test)


le.inverse_transform(2)