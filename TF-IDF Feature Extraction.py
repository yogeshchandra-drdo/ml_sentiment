
# coding: utf-8

# In[13]:


import pandas as pd

# read file into pandas using a relative path
path = "D:\ML Internship\sms.tsv"
sms = pd.read_table(path, header=None, names=['label', 'message'])


# In[14]:


sms.shape


# In[15]:


sms.head(10)


# In[16]:


# examine the class distribution
sms.label.value_counts()


# In[17]:


# convert label to a numerical variable because some classification algorithms want their classes to be of numeric values as well.
sms['label'] = sms.label.map({'ham':0, 'spam':1})


# In[18]:


# check that the conversion worked
sms.head(10)


# In[29]:


#Some preprocessing needs to be done before we start extracting features.

# Making everything lower case 
sms['message'] = sms['message'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[30]:


sms.head()


# In[31]:


# Removing Punctuation

sms['message'] = sms['message'].str.replace('[^\w\s]','')
sms['message'].head()


# In[33]:


# Removal of Stop Words

stop = set([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])

sms['message'] = sms['message'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
sms['message'].head()


# In[38]:


# how to define X and y (from the SMS data) for use with TF-IDFVectorizer
X = sms.message
y = sms.label
print(X.shape)
print(y.shape)


# In[39]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000)
train_vect = tfidf.fit_transform(X_train)

train_vect


# In[47]:


print(train_vect)


# In[48]:


test_vect = tfidf.fit_transform(X_test)

test_vect


# In[49]:


print(test_vect)


# In[50]:


# Now we have processed the text to make it useful for running further classification or Regression algorithms

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[54]:


# train the model using train_vect (timing it with an IPython "magic command")
get_ipython().run_line_magic('time', 'nb.fit(train_vect, y_train)')


# In[55]:


# make class predictions for test_vect
y_pred_class = nb.predict(test_vect)


# In[56]:


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[57]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[58]:


# print message text for the false positives (ham incorrectly classified as spam)
X_test[(y_pred_class==1) & (y_test==0)]
#or more elegant way is to use X_test[y_pred_class > y_test]


# In[59]:


# print message text for the false negatives (spam incorrectly classified as ham)
X_test[(y_pred_class==0) & (y_test==1)]
#or more elegant way is to use 
#X_test[y_pred_class < y_test]


# In[61]:


# Comparing models
#We will compare multinomial Naive Bayes with logistic regression:

# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[62]:


# train the model using X_train_dtm
get_ipython().run_line_magic('time', 'logreg.fit(train_vect, y_train)')


# In[63]:


# make class predictions for X_test_dtm
y_pred_class = logreg.predict(test_vect)


# In[64]:


# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)


# In[65]:


# Examining a model for further insight
#We will examine the our trained Naive Bayes model to calculate the approximate "spamminess" of each token.


# In[66]:


# store the vocabulary of X_train
X_train_tokens = tfidf.get_feature_names()
len(X_train_tokens)


# In[67]:


# examine the first 50 tokens
print(X_train_tokens[0:50])


# In[68]:


# examine the last 50 tokens
print(X_train_tokens[-50:])


# In[69]:


# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_


# In[70]:


# rows represent classes, columns represent tokens
nb.feature_count_.shape


# In[71]:


# number of times each token appears across all HAM messages
ham_token_count = nb.feature_count_[0, :]
ham_token_count


# In[72]:


# number of times each token appears across all SPAM messages
spam_token_count = nb.feature_count_[1, :]
spam_token_count


# In[76]:


# create a DataFrame of tokens with their separate ham and spam counts
tokens = pd.DataFrame({'token':X_train_tokens, 'ham':ham_token_count, 'spam':spam_token_count}).set_index('token')
tokens.sample(5,random_state=3)


# In[77]:


# Naive Bayes counts the number of observations in each class
nb.class_count_


# In[81]:


# add 1 to ham and spam counts to avoid mathematical errors like dividing by zero.
tokens['ham'] = tokens.ham + 1
tokens['spam'] = tokens.spam + 1
tokens.sample(5, random_state=3)


# In[82]:


# convert the ham and spam counts into frequencies
tokens['ham'] = tokens.ham / nb.class_count_[0]
tokens['spam'] = tokens.spam / nb.class_count_[1]
tokens.sample(5, random_state=3)


# In[83]:


# calculate the ratio of spam-to-ham for each token
tokens['spam_ratio'] = tokens.spam / tokens.ham
tokens.sample(5, random_state=3)


# In[84]:


# examine the DataFrame sorted by spam_ratio
# note: use sort() instead of sort_values() for pandas 0.16.2 and earlier
tokens.sort_values('spam_ratio', ascending=False)


# In[85]:


# look up the spam_ratio for a given token
tokens.loc['prize', 'spam_ratio']


# In[89]:


# look up the spam_ratio for a given token
tokens.loc['doing', 'spam_ratio']

