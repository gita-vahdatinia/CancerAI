import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns;  #makes graph colors pretty
from gensim.summarization import keywords
import os
import re
import string
import pandas as pd
import numpy as np
import keras
sns.set()
sns.palplot(sns.color_palette("muted"))


# get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_var = pd.read_csv("../input/training_variants")
test_var = pd.read_csv("../input/test_variants")
train_text = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
#dataframe.shape returns a tuple representing dimensionality
# print("Train and Test variants shape : ",train_var.shape, test_var.shape)
# print("Train and Test text shape : ",train_text.shape, test_text.shape)


# In[3]:


#dataframe.head returns first 5 rows
#training variants
train_var.head()
train_var.describe(include='all')


# In[4]:


#returns first 5 rows of training text
train_text.head()
train_text.describe(include='all')


# In[5]:


# train_data = pd.concat([train_var_df.set_index('ID'),
#                         train_text_df.set_index('ID')],axis=1)
train_data = pd.merge(train_var, train_text, how='left',on = 'ID').fillna('')
train_data['ID'] = train_data.ID.astype(int)
train_y = train_data['Class'].values
train_x = train_data.drop('Class',axis=1)
train_size=len(train_x)
train_data.head()


# In[6]:


train_data.describe(include='all')


# In[7]:


#dataframe.head returns first 5 rows
#test variants
test_var.head()


# In[8]:


#returns first 5 rows of test text
test_text.head()


# In[9]:


# test_data = pd.concat([test_var_df.set_index('ID'),
#                         test_text_df.set_index('ID')],axis=1)
test_data = pd.merge(test_var, test_text, how='left',on = 'ID').fillna('')
test_data['ID'] = test_data.ID.astype(int)
test_data.head()


# In[10]:


test_data.describe(include='all')


# In[11]:


#print 1st row from training text dataframe
with open("../input/training_text") as infile:
    for i in range(0,2):
        line = infile.readline()
        # print(line)


# In[12]:


#frequenct of classes in training data
# plt.figure(figsize=(12,8))
# sns.countplot(x="Class", data=train_var)
# plt.xlabel('Class Count', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.xticks(rotation='vertical')
# plt.title("Frequency of Classes", fontsize=15)
# plt.show()


# In[13]:


#First column is Class
#Second column is Frequency
#All in training variants file
data=train_var
data["Class"].value_counts()


# In[14]:


#print(train_data)
train_data['Variation'].describe()


# In[15]:


#drop columns gene and variation
#print(train_data)
#train_data.drop(['Gene', 'Variation'], axis=1, inplace=True)


# In[16]:


#https://rare-technologies.com/text-summarization-with-gensim/
from gensim.summarization import summarize
data_id = 0
#strictly gets the text from id 0
text = train_data.loc[data_id,'Text']
#print(text)
# print ('Summary:')
#could set a limit on how many words to return etc.
# print (summarize(text,split=True))


# In[17]:


# #algorithm tries to find words that are important
# #or seem representative of the entire text
# from gensim.summarization import keywords
# #lemmatisation determines the lemma of a word
# #based on its intended meaning
# trigger_words = keywords(text,words = 5,scores=True,lemmatize=True, split=True)
# print ("Keywords:")
# trigger_words =', '.join(['{}-{:.2f}'.format(i, j) for i, j in trigger_words])
# print ("["+trigger_words+"]")





# In[18]:


# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from string import punctuation
# from scipy.misc import imresize
# from PIL import Image
# from wordcloud import WordCloud, ImageColorGenerator
# from collections import Counter

# custom_words = ["fig", "figure", "et", "al", "al.", "also",
#                 "data", "analyze", "study", "table", "using",
#                 "method", "result", "conclusion", "author",
#                 "find", "found", "show", '"', "’", "“", "”"]

# stop_words = set(stopwords.words('english') + list(punctuation) + custom_words)
# wordnet_lemmatizer = WordNetLemmatizer()

# class_corpus = train_data.groupby('Class').apply(lambda x: x['Text'].str.cat())
# class_corpus = class_corpus.apply(lambda x: Counter(
#     [wordnet_lemmatizer.lemmatize(w)
#      for w in word_tokenize(x)
#      if w.lower() not in stop_words and not w.isdigit()]
# ))


# In[19]:


test_index = test_data['ID'].values
all_data = np.concatenate((train_x, test_data), axis=0)
all_data = pd.DataFrame(all_data)
all_data.columns = ["ID", "Gene", "Variation", "Text"]
all_data['ID'] = all_data.ID.astype(int)
# print (all_data.dtypes)
all_data.head()


# In[20]:


#figure out why there are nan objects?


# In[21]:


#https://dunkley.me/blog/msk-redefining-cancer-treatment

#did not lemmatize words use Spacy lemmatizer
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
import gensim

def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(TaggedDocument(gensim.utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    print (sentences[0])
    return sentences

def textClean(text):
    #only including alphanumeric symbols
    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"[()$!.\\\/'^+=<>?{}%&*;≤,\"'`_]"," ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return(text)

def cleanup(text):
    text = textClean(text)
    #gets rid of punctiations
    text= text.translate(str.maketrans("","", string.punctuation))
    return text

# all_data = all_data['Text'].str.decode("utf8")
#all_data['Text'] = all_data.Text.astype(str)
# print (all_data.dtypes)
#allText = textClean(str(all_data['Text'])
allText = all_data['Text'].apply(cleanup)
#sentences = constructLabeledSentences(allText)
allText.head()


# In[22]:


# print(train_data.dtypes)
# newtext = train_data['Text'].apply(cleanup)
# newtext.head()


# In[23]:


sentences = constructLabeledSentences(allText)
sentences[0]


# In[24]:

#
# from gensim.models import Doc2Vec
#
# Text_INPUT_DIM=300
#
#
# text_model=None
# filename='docEmbeddings_5_clean.d2v'
# if os.path.isfile(filename):
#     text_model = Doc2Vec.load(filename)
# else:
#     #doc2vec
#     #workers is the number of threads
#     #window
#     #using words you have to create to vectorize
#     #have all words in model and
#     #use that model to create the array
#     #window is distance between current and predicted word
#     #sample the threshold for configuring which higher-frequency words are randomly downsampled, useful range is
#     #negative specifies how many noise words should be drawn
#     #epochs (int) – Number of iterations (epochs) over the corpus.
#     text_model = Doc2Vec(min_count=1, window=5,vector_size=Text_INPUT_DIM, sample=1e-4, negative=5, workers=4, epochs=5,seed=1)
#     text_model.build_vocab(sentences)
#     text_model.train(sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)
#     text_model.save(filename)
#
#
# # In[25]:
#
#
# test_size =(len(test_data))
# text_train_arrays = np.zeros((train_size, Text_INPUT_DIM))
# text_test_arrays = np.zeros((test_size, Text_INPUT_DIM))
#
# #never train with test set so you split training data into
# #training and validaition sets
#
# #vectorization of training text
# for i in range(train_size):
#     text_train_arrays[i] = text_model.docvecs['Text_'+str(i)]
#
# j=0
#
# #vectorizing test data
# for i in range(train_size,train_size+test_size):
#     text_test_arrays[j] = text_model.docvecs['Text_'+str(i)]
#     j=j+1
#
# #vector for each id
# #3321 points in 300 dimensional space
# #text_train_arrays[0] is first vector
# #len(text_train_arrays[0]) = 300
# # 3321 indices
# #print("Size of text_train_arrays:")
#
# print (len(text_train_arrays))
# #print (len(text_train_arrays[0]))
# print(text_train_arrays[0])
# #print(text_train_arrays[0][:50])
#
#
# # In[26]:
#
#
# from sklearn.decomposition import TruncatedSVD
# #n_components is number of dimenstions you want
# Gene_INPUT_DIM=25
# #we use the new k-dimensional LSI representation as we did the original
# #representation – to compute similarities between vectors
#
# #one hot encoding is representation of categorical to binary
# #categoral values mapped to integer values
# #integer value represented as binary vector as all zeros except index of integer
# #
# svd = TruncatedSVD(n_components=25, n_iter=Gene_INPUT_DIM, random_state=12)
#
# #pd.get_dummies gets one hot encodings
# one_hot_gene = pd.get_dummies(all_data['Gene'])
# truncated_one_hot_gene = svd.fit_transform(one_hot_gene.values)
# #truncated = list of vectors 25 dimensions each vector representing a hot gene
# one_hot_variation = pd.get_dummies(all_data['Variation'])
# truncated_one_hot_variation = svd.fit_transform(one_hot_variation.values)
# print (one_hot_gene)
# print(truncated_one_hot_gene[0])
#
#
# # In[27]:
#
#
# from keras.utils import np_utils
# from sklearn.preprocessing import LabelEncoder
#
# #train_y = train['Class'].values
# #train_y is classes
# label_encoder = LabelEncoder()
# label_encoder.fit(train_y)
# encoded_y = np_utils.to_categorical((label_encoder.transform(train_y)))
# #print(encoded_y[0])
# print(encoded_y)
# print(len(encoded_y))
#
#
# # In[28]:
#
#
# #np.hstack concatenates arrays together
# #train_set is array of 300+25+25
# #vectorized one hot gene/variation/and text
#
# #same thing with test set
# #vectorized all important info
# train_set=np.hstack((truncated_one_hot_gene[:train_size],truncated_one_hot_variation[:train_size],text_train_arrays))
# test_set=np.hstack((truncated_one_hot_gene[train_size:],truncated_one_hot_variation[train_size:],text_test_arrays))
# print(len(text_train_arrays))
# #first 50 dimensions of first
# print(len(train_set[0]))
# print(train_set[0])
# #print(train_set[0][:50])
#
#
# # In[29]:
#
#
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
# from keras.optimizers import SGD
# #https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# #connecting each neuron to next neuron to next layer(Dense)
#
# #dense is connecting layers together
# #dropout - randomly draw connections between neurons
# #Text_INPUT_DIM=300 = input_dim = expected 300 variables
# #total input dimension is 350
# #fully connected layers are defined using the Dense class
# def baseline_model():
#     #sequental is linear stack of layers
#     #pass list of layers
#     model = Sequential()
#     model.add(Dense(256, input_dim=Text_INPUT_DIM+Gene_INPUT_DIM*2, init='normal', activation='relu')) #convolution layer 1
#     model.add(Dropout(0.3)) #max pooling - shrinks input
#     model.add(Dense(256, init='normal', activation='relu')) #convolution layer 2
#     model.add(Dropout(0.5))
#     model.add(Dense(80, init='normal', activation='relu')) #other link has 100 neurons, fully connected layer
#     model.add(Dense(9, init='normal', activation="softmax")) #output layer has 9 neurons classification output layer
#
#     #gradient descent optimizer
#     sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     return model
#
#
# # In[30]:
#
#
# model = baseline_model()
# model.summary()
#
#
# # In[31]:
#
#
# #validation for training accuracy
# #accuracy
# #validation
# #valuation loss
# #80% train data 20% validation data
# #validation
# estimator=model.fit(train_set, encoded_y, validation_split=0.2, epochs=10, batch_size=64)
#
#
# # In[32]:
#
#
# print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1]))
#
#
# # In[33]:
#
#
# import matplotlib.pyplot as plt
#
# # summarize history for accuracy
# plt.plot(estimator.history['acc'])
# plt.plot(estimator.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'valid'], loc='upper left')
# plt.show()
#
# # summarize history for loss
# plt.plot(estimator.history['loss'])
# plt.plot(estimator.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'valid'], loc='upper left')
# plt.show()
#
#
# # In[34]:
#
#
# y_pred = model.predict_proba(test_set)
#
#
# # In[35]:
#
#
# submission = pd.DataFrame(y_pred)
# submission['id'] = test_index
# submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
# submission.to_csv("submission_all.csv",index=False)
# submission.head()
#
#
# # In[36]:
#
#
# from keras import backend as K
# import seaborn as sns
#
# layer_of_interest=0
# intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[layer_of_interest].output])
# intermediate_tensor = intermediate_tensor_function([train_set[0,:].reshape(1,-1)])[0]
#
#
# # In[37]:
#
#
# import matplotlib
# colors = list(matplotlib.colors.cnames)
#
# intermediates = []
# color_intermediates = []
# for i in range(len(train_set)):
#     output_class = np.argmax(encoded_y[i,:])
#     intermediate_tensor = intermediate_tensor_function([train_set[i,:].reshape(1,-1)])[0]
#     intermediates.append(intermediate_tensor[0])
#     color_intermediates.append(colors[output_class])
#
#
# # In[38]:
#
#
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, random_state=0)
# intermediates_tsne = tsne.fit_transform(intermediates)
# plt.figure(figsize=(8, 8))
# plt.scatter(x = intermediates_tsne[:,0], y=intermediates_tsne[:,1], color=color_intermediates)
# plt.show()
