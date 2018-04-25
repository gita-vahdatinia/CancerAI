import gensim
import time

eTime = time.time()

filename = 'training_variants'
fp = open(filename)
text = fp.read()
#print(data)
lines = text.split('\n')
#print(lines)

data = []
for line in lines:
    data.append(line.split(','))


model = gensim.models.Word2Vec(data, min_count=1, seed=1, workers=1)#KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?

result = model.most_similar(negative=['FAM58A'], topn=1)
eTime = time.time() - eTime
print(result)
print("computation time: ", eTime)