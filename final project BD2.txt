import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

# Load data into python env
#we have created a small dataset form the bigger one and then analyzed
#For run the entire dataset please remove the # below and then upload the bigger dataset named:yelpreviews
#smaller dataset is named yelp.csv
#yelp.csv is recommended for lesser time.

#df =pd.read_csv('yelpreview.csv', nrows = 100000)
df = pd.read_csv('yelp.csv')

#these commands are used for checking the results. 

#shape() is for checking number of column and number of attributes
 
df.shape

#head() is used for viewing first 5 line of the dataset

df.head()

#info is used for information obout the dataset. example: memory size, non null object etc.

df.info()

#describe() tells you the mean, mediam, mode etc.

df.describe()


# Creating a new column and counting the text length

df['text length'] = df['text'].apply(len)
df.head()

#used seaborn library

#histogram
h = sns.FacetGrid(data=df, col='stars')
h.map(plt.hist, 'text length', bins=100)

#boxplot
sns.boxplot(x='stars', y='text length', data=df)


'''
group the data by the star rating, cool and funny 
and see if we can find a correlation between features such as cool, useful, and funny.
'''
stars = df.groupby('stars').mean()
stars.corr()

cool = df.groupby('cool').mean()
cool.corr()

funny = df.groupby('funny').mean()
funny.corr()

sns.heatmap(data=stars.corr(), annot=True)
sns.heatmap(data=cool.corr(), annot=True)
sns.heatmap(data=funny.corr(), annot=True)


#predict if a review is either bad or good
#only 5 stars and 1 stars are considered.

yelp_class = df[(df['stars'] == 1) | (df['stars'] == 5)]
yelp_class.shape


X = yelp_class['text']
y = yelp_class['stars']



#function for punctuation, stopwords in english language.

import string
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]




#Vectorisation in words

bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
len(bow_transformer.vocabulary_)


#let’s try a random review and get its bag-of-words counts as a vector.
#only try 5 star and 1 star reviews.

review_25 = X[24]
review_25

bow_25 = bow_transformer.transform([review_25])
bow_25

# finding the frequency of a perticular word.
temp = bow_25.todense()
temp
A = np.squeeze(np.asarray(temp))
indexes = np.nonzero(A)[0]
indexes

for items in indexes:
    print(items,A[items])



print(bow_transformer.get_feature_names()[11443])
print(bow_transformer.get_feature_names()[22077])

X = bow_transformer.transform(X)
#10 mins


print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print('Density: {}'.format((density)))


#Training data and test data
#Only 30% of test size from the dataset is considered.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Training our model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)


#Testing and evaluating our model
preds = nb.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))



#Data Bias 
#Predicting a singular positive review

positive_review = yelp_class['text'][59]
positive_review

positive_review_transformed = bow_transformer.transform([positive_review])
nb.predict(positive_review_transformed)[0]



#Predicting a singular negative review

negative_review = yelp_class['text'][281]
negative_review

negative_review_transformed = bow_transformer.transform([negative_review])
nb.predict(negative_review_transformed)[0]




#Where the model goes wrong

another_negative_review = yelp_class['text'][140]
another_negative_review

another_negative_transformed = bow_transformer.transform([another_negative_review])
nb.predict(another_negative_transformed)[0]
