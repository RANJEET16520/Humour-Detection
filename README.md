# Humour-Detection

### Description
#### Yelp-Review
Humour detection has an important role in applications like a chatbot, human-machine interaction. But automation of this task is complex due to the semantic structure of the text. In this project, we aim to use the yelp customer review dataset wherein each review has a weighted tag ’funny’. The dataset being crowd annotated is of high quality for the task of humour detection. Each review is tokenized followed by generating word vectors using word2vec. The resulting word vectors are used as input to the Convolutional Neural Network and Long Short-Term Memory. The Long Short-Term Memory had a high performance at the expense of training time. Long Short-Term Memory outperforms Convolutional Neural Network in all metrics used i.e. precision, recall, accuracy, F1 score, and area under ROC curve.

#### New 200K Jokes
In 200K Jokes, the each humourous and non-humourous category has 100K jokes. Every sample text is tokenized to words, and then word vectors are generated using Word2Vec. The word vectors are provided as input to XLNet Transformer and output is classified as 1 (for humourous or funny text) and 0 (for simple text).

### Data Collection
The Yelp-Review and New 200,000 Jokes Dataset is collected from Kaggle.

### Technologies Used
Deep Learning, Transfer Learning, NLP, NLP with Transformers.

#### Python Libraries
Numpy , MatplotLib , Keras, Tensorflow-GPU, NLTK, Seaborn, SkLearn.

##### Requirements
```
python 3

pip3 

Google Colab
```

### Publication

+ The Research Publication for this project has been submitted. 


### Contributors

[Ranjeet Walia](https://github.com/RANJEET16520)

[Shivam Sharma](https://github.com/shiv-7)




Thank you for visiting.

!! Don't forget to star.
