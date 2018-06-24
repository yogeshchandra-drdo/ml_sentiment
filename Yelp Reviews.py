
# coding: utf-8

# In[3]:


import pandas as pd

# read file into pandas using a relative path
path = "D:\ML\yelp_labelled.txt"
data = pd.read_table(path, header=None, names=['reviews', 'rating'])


# In[5]:


data.shape


# In[6]:


data.head(5)


# In[7]:


data.tail(5)


# In[8]:


data.rating.value_counts()


# In[9]:


#Creating data frames

X = data.reviews
y = data.rating

print(X.shape)
print(y.shape)


# In[10]:


X.head()


# In[11]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[13]:


#Part 4: Vectorizing our dataset

# import and instantiate the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# In[14]:


# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(X_train)


# In[15]:


#Creating document-term matrix
X_train_dtm = vect.transform(X_train)


# In[16]:


# examine the document-term matrix
X_train_dtm


# In[19]:


# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# In[38]:


#examine the vocabulary and document-term matrix together
pd.DataFrame(X_train_dtm.toarray(), columns=vect.get_feature_names())


# In[20]:


# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[21]:


# train the model using X_train_dtm (timing it with an IPython "magic command")
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')


# In[22]:


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# In[23]:


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[24]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[ ]:


nb.predict_proba(X_test_dtm)


# In[27]:


# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# In[85]:


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# In[29]:


#Part 2: Comparing models
#We will compare multinomial Naive Bayes with logistic regression:

# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[30]:


# train the model using X_train_dtm
get_ipython().run_line_magic('time', 'logreg.fit(X_train_dtm, y_train)')


# In[31]:


# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)


# In[32]:


# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)


# In[33]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[34]:


# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# In[35]:


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# In[39]:


#Part 3: Examining a model for further insight
#We will examine the our trained Naive Bayes model to calculate the approximate "goodness" of each token.


# In[40]:


# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


# In[41]:


# examine the first 50 tokens
print(X_train_tokens[0:50])


# In[42]:


# examine the last 50 tokens
print(X_train_tokens[-50:])


# In[43]:


# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_


# In[44]:


# rows represent classes, columns represent tokens
nb.feature_count_.shape


# In[51]:


# number of times each token appears across all HAM messages
bad_token_count = nb.feature_count_[0, :]
bad_token_count


# In[52]:


# number of times each token appears across all SPAM messages
good_token_count = nb.feature_count_[1, :]
good_token_count


# In[74]:


# create a DataFrame of tokens with their separate ham and spam counts
tokens = pd.DataFrame({'token':X_train_tokens, 'bad':bad_token_count, 'good':good_token_count}).set_index('token')
tokens.head()


# In[75]:


# examine 5 random DataFrame rows
tokens.sample(5, random_state=5)


# In[76]:


# Naive Bayes counts the number of observations in each class
nb.class_count_


# In[77]:


# add 1 to ham and spam counts to avoid mathematical errors like dividing by zero.
tokens['bad'] = tokens.bad + 1
tokens['good'] = tokens.good + 1
tokens.sample(5, random_state=5)


# In[78]:


# convert the bad and good counts into frequencies
tokens['bad'] = tokens.bad / nb.class_count_[0]
tokens['good'] = tokens.good / nb.class_count_[1]
tokens.sample(5, random_state=5)


# In[79]:


# calculate the ratio of good-to-bad for each token
tokens['good_ratio'] = tokens.good / tokens.bad
tokens.sample(5, random_state=5)


# In[81]:


# examine the DataFrame sorted by good_ratio
# note: use sort() instead of sort_values() for pandas 0.16.2 and earlier
tokens.sort_values('good_ratio', ascending=False)


# In[84]:


# look up the good_ratio for a given token
tokens.loc['great', 'good_ratio']

