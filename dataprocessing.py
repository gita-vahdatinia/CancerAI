
# coding: utf-8

# In[18]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns;  #makes graph colors pretty
from gensim.summarization import keywords
#from __future__ import print_function
import os
import re
import string
import pandas as pd
import numpy as np
import keras
from sklearn.manifold import TSNE
import matplotlib
sns.set()
sns.palplot(sns.color_palette("muted"))


#get_ipython().magic(u'matplotlib inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



# In[19]:


train_var = pd.read_csv("../input/training_variants")
test_var = pd.read_csv("../input/stage2_test_variants.csv")
train_text = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("../input/stage2_test_text.csv", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
#dataframe.shape returns a tuple representing dimensionality
#print("Train and Test variants shape : ",train_var.shape, test_var.shape)
#print("Train and Test text shape : ",train_text.shape, test_text.shape)


# In[20]:


#dataframe.head returns first 5 rows
#training variants
train_var.head()
train_var.describe(include='all')


# In[21]:


#returns first 5 rows of training text
train_text.head()
train_text.describe(include='all')


# In[22]:


# train_data = pd.concat([train_var_df.set_index('ID'),
#                         train_text_df.set_index('ID')],axis=1)
train_data = pd.merge(train_var, train_text, how='left',on = 'ID').fillna('')
train_data['ID'] = train_data.ID.astype(int)
train_y = train_data['Class'].values
train_x = train_data.drop('Class',axis=1)
train_size=len(train_x)
train_data.head()


# In[23]:


train_data.describe(include='all')


# In[24]:


#dataframe.head returns first 5 rows
#test variants
#test_var.head()


# In[25]:


#returns first 5 rows of test text
#test_text.head()


# In[26]:


# test_data = pd.concat([test_var_df.set_index('ID'),
#                         test_text_df.set_index('ID')],axis=1)
test_data = pd.merge(test_var, test_text, how='left',on = 'ID').fillna('')
test_data['ID'] = test_data.ID.astype(int)
test_data.head()


# In[27]:


test_data.describe(include='all')


# In[28]:


#print 1st row from training text dataframe
# with open("../input/training_text") as infile:
#     for i in range(0,2):
#         line = infile.readline()
#         print(line)


# In[29]:


#frequenct of classes in training data
plt.figure(figsize=(12,8))
sns.countplot(x="Class", data=train_var)
plt.xlabel('Class Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Classes", fontsize=15)
#plt.show()


# In[30]:


#First column is Class
#Second column is Frequency
#All in training variants file
data=train_var
data["Class"].value_counts()      


# In[31]:


#print(train_data)
train_data['Variation'].describe()


# In[32]:


test_index = test_data['ID'].values
all_data = np.concatenate((train_x, test_data), axis=0)
all_data = pd.DataFrame(all_data)
all_data.columns = ["ID", "Gene", "Variation", "Text"]
all_data['ID'] = all_data.ID.astype(int)
#print (all_data.dtypes)
#all_data.head()


# In[ ]:


#https://dunkley.me/blog/msk-redefining-cancer-treatment
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import spacy

from gensim.models.doc2vec import TaggedDocument
from gensim import utils
import gensim
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS

custom_words = ["fig", "figure", "et", "al", "al.", "also",
                "data", "analyze", "study", "table", "using",
                "method", "result", "conclusion", "author", 
                "find", "found", "show","them",'study','case','syndrome', 
                'author', 'show', 'control', 'expression','supplementary',
                'result', 'figure','fig', 'level', 'deletion', 'mm',
                'state', 'effect', 'stability', 'activity','change','structure',
                'line', 'loss', 'expression' '"', "’", "“", "”", "disease",
                "diseases", "disorder", "symptom", "symptoms", "drug", "drugs",
                "problems", "problem","prob", "probs", "med", "meds",
                "pill", "pills", "medicine", "medicines", "medication", "medications",
                "treatment", "treatments", "caps", "capsules", "capsule",
                "tablet", "tablets", "tabs", "doctor", "dr"," dr."," doc",
                "physician", "physicians", "test", "tests", "testing",
                "specialist", "specialists","side-effect", "side-effects", 
                "pharmaceutical", "pharmaceuticals", "pharma", "diagnosis",
                "diagnose", "diagnosed"," exam","challenge", "device", "condition",
                "conditions", "suffer", "suffering" ,"suffered", "feel"," feeling",
                "prescription", "prescribe","prescribed", "over-the-counter", "otc"]

def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(TaggedDocument(gensim.utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

#print("Pre sentence",sentences[0])

def textClean(text):
    #only including alphanumeric symbols
  #  text = re.sub(r"[^A-Za-z0-9^,!.\/'+=]", "", text)
    #text = re.sub(r'\([^)]*\)', "", text)
   # text = re.sub(r"[()$!/\^.\/'^+=<>?{}%&*;',\:]", " ", text)
    text = re.sub(r"[()$!.\/'^+=<>?{}%&*;',:]", " ", text)
    text = text.lower().split()
#    print("textClean", text)
    stops = set(stopwords.words("english") + custom_words)
    wordnet_lemmatizer = WordNetLemmatizer()
    text = [wordnet_lemmatizer.lemmatize(w) for w in text if not w in stops and w.isalpha() and stop_words.ENGLISH_STOP_WORDS and STOP_WORDS] 
    text = " ".join(text)
    return(text)
    
def cleanup(text):
    text = textClean(text)
    #print("HIIII",text)
    #gets rid of punctuations
   # text= text.translate(str.maketrans("","", string.punctuation))

    
    return text

allText = all_data['Text'].apply(cleanup)
allText.head()






# In[ ]:


sentences = constructLabeledSentences(allText)
sentences[0]


# In[ ]:


from gensim.models import Doc2Vec

Text_INPUT_DIM=300


text_model=None
filename='docEmbeddings_5_clean.d2v'
if os.path.isfile(filename):
    text_model = Doc2Vec.load(filename)
else:
    text_model = Doc2Vec(min_count=1, window=5,vector_size=Text_INPUT_DIM, sample=1e-4, negative=5, workers=4, epochs=5,seed=1)
    text_model.build_vocab(sentences)
    text_model.train(sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)
    text_model.save(filename)


# In[ ]:


test_size =(len(test_data))
text_train_arrays = np.zeros((train_size, Text_INPUT_DIM))
text_test_arrays = np.zeros((test_size, Text_INPUT_DIM))

for i in range(train_size):
    text_train_arrays[i] = text_model.docvecs['Text_'+str(i)]

j=0

#vectorizing test data
for i in range(train_size,train_size+test_size):
    text_test_arrays[j] = text_model.docvecs['Text_'+str(i)]
    j=j+1

# In[ ]:


from sklearn.decomposition import TruncatedSVD

Gene_INPUT_DIM=25

svd = TruncatedSVD(n_components=25, n_iter=Gene_INPUT_DIM, random_state=12)

#pd.get_dummies gets one hot encodings
one_hot_gene = pd.get_dummies(all_data['Gene'])
truncated_one_hot_gene = svd.fit_transform(one_hot_gene.values)
#truncated = list of vectors 25 dimensions each vector representing a hot gene
one_hot_variation = pd.get_dummies(all_data['Variation'])
truncated_one_hot_variation = svd.fit_transform(one_hot_variation.values)

# In[ ]:


from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

#train_y = train['Class'].values
#train_y is classes
label_encoder = LabelEncoder()
label_encoder.fit(train_y)
encoded_y = np_utils.to_categorical((label_encoder.transform(train_y)))
#print(encoded_y[0])
#print(encoded_y)
#print(len(encoded_y))


# In[ ]:


#np.hstack concatenates arrays together
#train_set is array of 300+25+25
#vectorized one hot gene/variation/and text

#same thing with test set
#vectorized all important info
train_set=np.hstack((truncated_one_hot_gene[:train_size],truncated_one_hot_variation[:train_size],text_train_arrays))
test_set=np.hstack((truncated_one_hot_gene[train_size:],truncated_one_hot_variation[train_size:],text_test_arrays))
#print(len(text_train_arrays))
#first 50 dimensions of first 
#print(len(train_set[0]))
#print(train_set[0])
#print(train_set[0][:50])


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.optimizers import SGD
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#connecting each neuron to next neuron to next layer(Dense)

#dense is connecting layers together
#dropout - randomly draw connections between neurons
#Text_INPUT_DIM=300 = input_dim = expected 300 variables
#total input dimension is 350
#fully connected layers are defined using the Dense class
def baseline_model():
    #sequental is linear stack of layers
    #pass list of layers
    model = Sequential()
    model.add(Dense(256, input_dim=Text_INPUT_DIM+Gene_INPUT_DIM*2, init='normal', activation='relu')) #convolution layer 1
    model.add(Dropout(0.5)) #max pooling - shrinks input 
    model.add(Dense(256, init='normal', activation='relu')) #convolution layer 2
    model.add(Dropout(0.5)) #max pooling - shrinks input 
#     model.add(Dense(80, init='normal', activation='relu')) #other link has 100 neurons, fully connected layer
    model.add(Dense(9, init='normal', activation="softmax")) #output layer has 9 neurons classification output layer
    #gradient descent optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[ ]:


model = baseline_model()
model.summary()


# In[ ]:


#validation for training accuracy
#accuracy 
#validation
#valuation loss
#80% train data 20% validation data
#validation 
estimator=model.fit(train_set, encoded_y, validation_split=0.2, epochs=80, batch_size=512)


# In[ ]:


print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1]))


# In[ ]:


import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(estimator.history['acc'])
plt.plot(estimator.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(estimator.history['loss'])
plt.plot(estimator.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[ ]:


y_pred = model.predict_proba(test_set)


# In[ ]:


submission = pd.DataFrame(y_pred)
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
submission = submission[['id', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']]


# In[ ]:


submission.to_csv("submission_all.csv",index=False)
submission.head()
#print(submission)


# In[ ]:


from keras import backend as K
import seaborn as sns

layer_of_interest=0
intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[layer_of_interest].output])
intermediate_tensor = intermediate_tensor_function([train_set[0,:].reshape(1,-1)])[0]


# In[ ]:

colors = list(matplotlib.colors.cnames)

intermediates = []
color_intermediates = []
for i in range(len(train_set)):
    output_class = np.argmax(encoded_y[i,:])
    intermediate_tensor = intermediate_tensor_function([train_set[i,:].reshape(1,-1)])[0]
    intermediates.append(intermediate_tensor[0])
    color_intermediates.append(colors[output_class])


# In[ ]:
tsne = TSNE(n_components=2, random_state=0)
intermediates_tsne = tsne.fit_transform(intermediates)
plt.figure(figsize=(8, 8))
plt.scatter(x = intermediates_tsne[:,0], y=intermediates_tsne[:,1], color=color_intermediates)
plt.show()

