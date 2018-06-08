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
sns.set()
sns.palplot(sns.color_palette("muted"))



pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_var = pd.read_csv("../input/training_variants")
test_var = pd.read_csv("../input/stage2_test_variants.csv")
train_text = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("../input/stage2_test_text.csv", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
#dataframe.shape returns a tuple representing dimensionality
#print("Train and Test variants shape : ",train_var.shape, test_var.shape)
#print("Train and Test text shape : ",train_text.shape, test_text.shape)

# train_data = pd.concat([train_var_df.set_index('ID'),
#                         train_text_df.set_index('ID')],axis=1)
train_data = pd.merge(train_var, train_text, how='left',on = 'ID').fillna('')
train_data['ID'] = train_data.ID.astype(int)
train_y = train_data['Class'].values
train_x = train_data.drop('Class',axis=1)
train_size=len(train_x)
train_data.head()

test_data = pd.merge(test_var, test_text, how='left',on = 'ID').fillna('')
test_data['ID'] = test_data.ID.astype(int)
test_data.head()
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

test_index = test_data['ID'].values
all_data = np.concatenate((train_x, test_data), axis=0)
all_data = pd.DataFrame(all_data)
all_data.columns = ["ID", "Gene", "Variation", "Text"]
all_data['ID'] = all_data.ID.astype(int)

custom_words = ["fig", "figure", "et", "al", "al.", "also",
                "data", "analyze", "study", "table", "using",
                "method", "result", "conclusion", "author", 
                "find", "found", "show","them",'study','case','syndrome', 
                'author', 'show', 'control', 'expression','supplementary',
                'result', 'figure','fig', 'level', 'deletion', 'mm',
                'state', 'effect', 'stability', 'activity','change','structure',
                'line', 'loss', 'expression' '"', "'", """, """, "disease",
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
    #text = re.sub(r"[^A-Za-z0-9^,!.\/'+=]", "", text)
    #text = re.sub(r'\([^)]*\)', "", text)
  #  text = re.sub(r"[()$!.\/'^+=<>?{}%&*;',:]", " ", text)
    text = re.sub(r"[()$!.\\\/'^+=<>?{}%&*;,\"'`]"," ", text)
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
#    text= text.translate(str.maketrans("","", string.punctuation))

    
    return text

allText = all_data['Text'].apply(cleanup)
print("AFTER:ALLTEXT:", allText)
#allText.head()

sentences = constructLabeledSentences(allText)
print(sentences[0])

