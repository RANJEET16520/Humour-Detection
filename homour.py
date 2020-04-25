import sys
print(sys.executable)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 



funny = []
nfunny = []
nfunny_small = []

with open('Raw_Data/funny.txt', 'rb') as file:
	funny = pickle.load(file)
print(len(funny))

with open('Raw_Data/nfunny_small.txt', 'rb') as file:
	nfunny_small = pickle.load(file)
print(len(nfunny_small))



contractions = { 
"n't": "not",
"'ve": "have",
"'cause": "because",
"'ll": "will",
"'re":"are",
"'m":"am"
}
contractions_list = ["n't", "'ve", "'cause", "'ll", "'re","'m"]

def tokenise(text):
	token = [i.replace('\n' , ' ') for i in sent_tokenize(text)]
	words = [word_tokenize(x.strip()) for x in token]
	word = []
	for s in words:
		if len(s) > 0:
			word.append(i.lower().split() for i in s)
	l = []
	for i in words:
		for j in i:
			if j in contractions_list:
				j = contractions[j]
				l.append(j)
				continue
			l.append(j.lower().strip())
	return l

def All_token(data):
	sentences = []
	for i in data:
		tokens = tokenise(i)
		sentences.append(tokens)
	return sentences

def RemoveStopWords(tokens):
	stop_words = set(stopwords.words('english')) 
	
	filtered_sentence = [] 
	for lis in tokens:
		w_lis = []
		for w in lis: 
			if w not in stop_words: 
				w_lis.append(w)
		filtered_sentence.append(w_lis)      
	return filtered_sentence

def CreateSentence(word_list):
	data = []
	for i in range(len(word_list)):
		every_word_list = word_list[i]
		sentence = ' '.join(word for word in every_word_list)
		data.append(sentence)
	return data

print("############################## Funny Data ##############################")
f_tokens = All_token(funny)
print(len(f_tokens))

filter_f_token = RemoveStopWords(f_tokens)
print(len(filter_f_token))

funny_data = CreateSentence(filter_f_token)
print(len(funny_data))

with open('Model_Input/funny_data.txt', 'wb') as fp:
	pickle.dump(funny_data, fp)

# Example
print(funny[1])
print(len(f_tokens[1]))
print(f_tokens[1])
print(len(filter_f_token[1]))
print(filter_f_token[1])
print(len(set(filter_f_token[1])))
print(set(filter_f_token[1]))
print(funny_data[1])

print("############################## Non-Funny Data ##############################")
nf_tokens = All_token(nfunny_small)
print(len(nf_tokens))

filter_nf_token = RemoveStopWords(nf_tokens)
print(len(filter_nf_token))

nfunny_data = CreateSentence(filter_nf_token)
print(len(nfunny_data))

with open('Model_Input/nfunny_data.txt', 'wb') as fo:
	pickle.dump(nfunny_data, fo)

# Example
print(nfunny_small[1])
print(len(nf_tokens[1]))
print(nf_tokens[1])
print(len(filter_nf_token[1]))
print(filter_nf_token[1])
print(len(set(filter_nf_token[1])))
print(set(filter_nf_token[1]))
print(nfunny_data[1])

print("############################## Word2Vec-Model ##############################")
import gensim   
from sklearn.decomposition import PCA

final_f_list = filter_f_token + filter_nf_token
print(len(final_f_list))

num_features = 100    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

w2v_model = gensim.models.Word2Vec(final_f_list, workers=num_workers,size=num_features, min_count = min_word_count, window = context, sample = downsampling)

words = list(w2v_model.wv.vocab)
print(len(words))
print(words)

print(w2v_model.similarity('sauce','modern'))
print(w2v_model['took'])
print(w2v_model['?'])
result = w2v_model.most_similar(positive=['queen', 'man'], negative=['king'], topn=1)
print(result)

Embedding_index = w2v_model[words]

pca = PCA(n_components=2)
result = pca.fit_transform(Embedding_index)
plt.scatter(result[:, 0], result[:, 1])
words = list(w2v_model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()

w2v_model.save("word2vec.model")
