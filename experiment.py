import gensim
import time
from collections import defaultdict, Counter
from statistics import mode
eTime = time.time()

filename = 'training_variants'
fp = open(filename)
text = fp.read()
#print(data)
lines = text.split('\n')
#print(lines)

data = []
for line in lines[1:-1]:
    data.append(line.split(','))

#model = gensim.models.Word2Vec(data, min_count=1, seed=1, workers=1)#KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?

#result = model.most_similar(negative=['FAM58A'], topn=1)
#print(result)

#print(data)
genes = defaultdict(list)
variations = defaultdict(list)
geneStats = defaultdict(list)
variationStats = defaultdict(list)

for mutation in data:
    genes[mutation[1]].append(mutation[3])
    variations[mutation[2]].append(mutation[3])

#print(genes)
for gene in genes:
    #print(len(genes[gene]))

    count = Counter(genes[gene])
    geneStats[gene] =   [count.most_common(2), len(genes[gene])]
for variation in variations:
    #print(variations[variation])

    count = Counter(variations[variation])
    variationStats[variation] = [count.most_common(2), len(variations[variation])]
print("genes:")
print(geneStats)
print("variations:")
print(variationStats)








eTime = time.time() - eTime
print("computation time: ", eTime)