#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[3]:


#csv = 'clean_tweet.csv'
#my_df = pd.read_csv(csv,index_col=0)


csv = "corpus/train_clean4.txt"
my_df = pd.read_csv(csv,index_col=0, sep="\t")

my_df.head()


# In[4]:


my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df.info()


# In[5]:


x = my_df.text#[:5000]
y = my_df.target#[:5000]


#N = 60000
#xt = []
#yt = []
#i = 0
#j=0
#while(j<N):
#    if y[i] == 0:
#        xt.append(x[i])
#        yt.append(y[i])
#        j = j+1
#    i = i+1
#i=800000
#while (j<(2*N)):
#    if y[i] == 1:
#        xt.append(x[i])
#        yt.append(y[i])
#        j = j+1
#    i = i+1#
#
#x = xtdhigs
#y = yt

# In[7]:


from sklearn.model_selection import train_test_split
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)


# In[8]:


print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                             (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                            (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
                                                                             (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
                                                                            (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                             (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                            (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))


# ## Aritificial Neural Network

# My first idea was, if logistic regression is the best performing classifier, then this idea can be extended to neural networks. In terms of its structure, logistic regression can be thought as a neural network with no hidden layer, and just one output node.

# ![title](img_02/lr_nn.png)

# I will not go through the details of how neural networks work, but if you want to know more in detail, I have written a post before on implementing a neural network from scratch with Python. But for this post, I won't implement it from scratch but use a library called Keras. Keras is more of a wrapper, which can be run on top of other libraries such as Theano or TensorFlow. It is one of the most easy-to-use libraries with intuitive syntax yet powerful. If you are a newbie to neural network modelling as myself, I think Keras is the way to go.

# ### ANN with Tfidf vectorizer

# The best performing Tfidf vectors I got is with 100,000 features including up to trigram with logistic regression. Validation accuracy is 82.91%, while train set accuracy is 84.19%. I would want to see if a neural network can boost the performance of my existing Tf-Idf vectors.

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
tvec1 = TfidfVectorizer(max_features=1000,ngram_range=(1, 3))
tvec1.fit(x_train)


# In[12]:


x_train_tfidf = tvec1.transform(x_train)


# In[13]:


x_validation_tfidf = tvec1.transform(x_validation).toarray()


# In[16]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train_tfidf, y_train)


# In[17]:


clf.score(x_validation_tfidf, y_validation)


# In[26]:


clf.score(x_train_tfidf, y_train)

print("ok0")

# I will first start by loading required dependencies. In order to run Keras with TensorFlow backend, you need to install both TensorFlow and Keras.

# In[33]:


seed = 7
np.random.seed(seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# The structure of below NN model has 100,000 nodes in the input layer, then 64 nodes in a hidden layer with Relu activation function applied, then finally one output layer with sigmoid activation function applied. There are different types of optimizing techniques for neural networks, and different loss function you can define with the model. Below model uses ADAM optimizing, and binary cross entropy loss.

# ADAM is an optimization algorithm for updating the parameters and minimizing the cost of the neural network, which is proved to be very effective. It combines two methods of optimization: RMSProp, Momentum. Again, I will focus on sharing the result I got from my implementation, but if you want to understand properly how ADAM works, I strongly recommend the "deeplearning.ai" course by Andrew Ng. He explains the complex concept of neural network in a very intuitive way.

# Before I feed the data and train the model, I need to deal with one more thing. Keras NN model cannot handle sparse matrix directly. The data has to be dense array or matrix, but transforming the whole training data Tfidf vectors of 1.5 million to dense array won't fit into my RAM. So I had to define a function, which generates iterable generator object so that it can be fed to NN model. Note that the output should be a generator class object rather than arrays, this can be achieved by using "yield" instead of "return". 

# In[18]:


def batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].toarray()
        y_batch = y_data[y_data.index[index_batch]]
        counter += 1
        yield X_batch,y_batch
        if (counter > number_of_batches):
            counter=0


# In[20]:


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=1000))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit_generator(generator=batch_generator(x_train_tfidf, y_train, 32),
                    epochs=5, validation_data=(x_validation_tfidf, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)


# It looks like the model had the best validation accuracy after 2 epochs, and after that, it fails to generalize so validation accuracy slowly decreases, while training accuracy increases. But if you remember the result I got from logistic regression (train accuracy: 84.19%, validation accuracy: 82.91%), you can see that the above neural network failed to outperform logistic regression in terms of validation.

# Let's see if normalizing inputs have any effect on the performance.

# In[42]:

print ("ok1")
from sklearn.preprocessing import Normalizer
norm = Normalizer().fit(x_train_tfidf)
x_train_tfidf_norm = norm.transform(x_train_tfidf)
x_validation_tfidf_norm = norm.transform(x_validation_tfidf)


# In[46]:


model_n = Sequential()
model_n.add(Dense(64, activation='relu', input_dim=100000))
model_n.add(Dense(1, activation='sigmoid'))
model_n.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

model_n.fit_generator(generator=batch_generator(x_train_tfidf_norm, y_train, 32),
                      epochs=5, validation_data=(x_validation_tfidf_norm, y_validation),
                      steps_per_epoch=x_train_tfidf_norm.shape[0]/32)

# By the look of the result, normalizing seems to have almost no effect on the performance. And it is at this point I realized that Tfidf is already normalized by the way it is calculated. TF (Term Frequency) in Tfidf is not absolute frequency but relative frequency, and by multiplying IDF (Inverse Document Frequency) to the relative term frequency value, it further normalizes the value in a cross-document manner.

# If the problem of the model is a poor generalization, then there is another thing I can add to the model. Even though the neural network is a very powerful model, sometimes overfitting to the training data can be a problem. Dropout is a technique that addresses this problem. If you are familiar with the concept of ensemble model in machine learning, dropout can also be seen in the same vein. According to the research paper "Improving neural networks by preventing
# co-adaptation of feature detectors" by Hinton et al. (2012), "A good way to reduce the error on the test set is to
# average the predictions produced by a very large number of different networks. The standard way to do this is to train many separate networks and then to apply each of these networks to the test data, but this is computationally expensive during both training and testing. Random dropout makes it possible to train a huge number of different networks in a reasonable time." https://arxiv.org/pdf/1207.0580.pdf

# Dropout is simulating as if we train many different networks and averaging them by randomly omitting hidden nodes with a certain probability throughout the training process. With Keras, this can be easily implemented just by adding one line to your model architecture. Let's see how the model performance changes with 20% dropout rate. (*I will gather all the results and present them with a table at the end.)

# In[21]:

print ("ok2")
model1 = Sequential()
model1.add(Dense(64, activation='relu', input_dim=100000))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model1.fit_generator(generator=batch_generator(x_train_tfidf, y_train, 32),
                    epochs=5, validation_data=(x_validation_tfidf, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)


# Through 5 epochs, the train set accuracy didn't get as high as the model without dropout, but validation accuracy didn't drop as low as the previous model. Even though the dropout added some generalization to the model, but the validation accuracy is still underperforming compared to logistic regression result.

# There is another method I can try to prevent overfitting. By presenting the data in the same order for every epoch, there's a possibility that the model learns the parameters which also includes the noise of the training data, which eventually leads to overfitting. This can be improved by shuffling the order of the data we feed the model. Below I added shuffling to the batch generator function and tried with the same model structure and compared the result.

# In[23]:


def batch_generator_shuffle(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    np.random.shuffle(index)
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].toarray()
        y_batch = y_data[y_data.index[index_batch]]
        counter += 1
        yield X_batch,y_batch
        if (counter > number_of_batches):
            np.random.shuffle(index)
            counter=0


# In[24]:


model_s = Sequential()
model_s.add(Dense(64, activation='relu', input_dim=100000))
model_s.add(Dense(1, activation='sigmoid'))
model_s.compile(optimizer='adam',       loss='binary_crossentropy',              metrics=['accuracy'])
model_s.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32), epochs=5, validation_data=(x_validation_tfidf, y_validation),steps_per_epoch=x_train_tfidf.shape[0]/32)


# The same model with non-shuffled training data had training accuracy of 87.36%, and validation accuracy of 79.78%. With shuffling, training accuracy decreased to 84.80% but the validation accuracy after 5 epochs has increased to 82.61%. It seems like the shuffling did improve the model's performance on the validation set. And another thing I noticed is that with or without shuffling also for both with or without dropout, validation accuracy tends to peak after 2 epochs, and gradually decrease afterwards.

# Below I tried the same model with 20% dropout with shuffled data, this time only 2 epochs.

# In[25]:


model_s_1 = Sequential()
model_s_1.add(Dense(64, activation='relu', input_dim=100000))
model_s_1.add(Dropout(0.2))
model_s_1.add(Dense(1, activation='sigmoid'))
model_s_1.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
model_s_1.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32),
                        epochs=5, validation_data=(x_validation_tfidf, y_validation),
                        steps_per_epoch=x_train_tfidf.shape[0]/32)


# As same as the non-shuffled data, both the training accuracy and validation accuracy slightly dropped.

# As I was going through the "deeplearning.ai" course by Andrew Ng, he states that the first thing he would try to improve a neural network model is tweaking the learning rate. I decided to follow his advice and try different learning rates with the model. Please note that except for the learning rate, the parameter for 'beta_1', 'beta_2', and 'epsilon' are set to the default values presented by the original paper "ADAM: A Method for Stochastic Optimization" by Kingma and Ba (2015). https://arxiv.org/pdf/1412.6980.pdf

# In[36]:


import keras
custom_adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model_testing_2 = Sequential()
model_testing_2.add(Dense(64, activation='relu', input_dim=100000))
model_testing_2.add(Dense(1, activation='sigmoid'))
model_testing_2.compile(optimizer=custom_adam,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
model_testing_2.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32),
                              epochs=2, validation_data=(x_validation_tfidf, y_validation),
                              steps_per_epoch=x_train_tfidf.shape[0]/32)


# In[37]:


custom_adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model_testing_3 = Sequential()
model_testing_3.add(Dense(64, activation='relu', input_dim=100000))
model_testing_3.add(Dense(1, activation='sigmoid'))
model_testing_3.compile(optimizer=custom_adam,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

model_testing_3.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32),
                              epochs=2, validation_data=(x_validation_tfidf, y_validation),
                              steps_per_epoch=x_train_tfidf.shape[0]/32)



custom_adam = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model_testing_4 = Sequential()
model_testing_4.add(Dense(64, activation='relu', input_dim=100000))
model_testing_4.add(Dense(1, activation='sigmoid'))
model_testing_4.compile(optimizer=custom_adam,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

model_testing_4.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32),
                              epochs=2, validation_data=(x_validation_tfidf, y_validation),
                              steps_per_epoch=x_train_tfidf.shape[0]/32)





custom_adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model_testing_5 = Sequential()
model_testing_5.add(Dense(64, activation='relu', input_dim=100000))
model_testing_5.add(Dense(1, activation='sigmoid'))
model_testing_5.compile(optimizer=custom_adam,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

model_testing_5.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32),
                              epochs=5, validation_data=(x_validation_tfidf, y_validation),
                              steps_per_epoch=x_train_tfidf.shape[0]/32)

# Having tried four different learning rates (0.0005, 0.005, 0.01, 0.1), none of them outperformed the default learning rate of 0.001.

# Maybe I can try to increase the number of hidden nodes, and see how it affects the performance. Below model has 128 nodes in the hidden layer.

# In[27]:


model_s_2 = Sequential()
model_s_2.add(Dense(128, activation='relu', input_dim=100000))
model_s_2.add(Dense(1, activation='sigmoid'))
model_s_2.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model_s_2.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32),
                        epochs=2, validation_data=(x_validation_tfidf, y_validation),
                        steps_per_epoch=x_train_tfidf.shape[0]/32)

print ("ok10")
# With 128 hidden nodes, validation accuracy got close to the performance of logistic regression. I could experiment further with increasing the number of hidden layers, but for the above 2 epochs to run, it took 5 hours. Considering that logistic regression took less than a minute to fit, even if the neural network can be improved further, this doesn't look like an efficient way.

# Below is a table with all the results I got from trying different models above. Please note that I have compared performance at 2 epochs since some of the models only ran for 2 epochs.

# | model | learning rate | input layer (nodes) | data shuffling | hidden layer (nodes) | dropout | output layer (nodes) | training accuracy | validation accuracy |
# |-----|------|---|-----|----|----|----|----|
# | ANN_1 | 0.001 | 1 (100,000)  | X | 1 (64) relu  |  X  | 1 (1) sigmoid   | 83.52% | 82.54% |
# | ANN_2 | 0.001 | 1 (100,000)  | X | 1 (64) relu  |  0.2  | 1 (1) sigmoid   | 83.35% | 82.56% |
# | ANN_3 | 0.001 | 1 (100,000)  | O | 1 (64) relu  |  X  | 1 (1) sigmoid   | 83.52% | 82.76% |
# | ANN_4 | 0.001 | 1 (100,000)  | O | 1 (64) relu  |  0.2  | 1 (1) sigmoid   | 83.37% | 82.64%  |
# | ANN_5 | 0.0005 | 1 (100,000)  | O | 1 (64) relu  |  X  | 1 (1) sigmoid   | 83.52% | 82.61%  |
# | ANN_6 | 0.005 | 1 (100,000)  | O | 1 (64) relu  |  X  | 1 (1) sigmoid   | 83.52% | 82.59%  |
# | ANN_7 | 0.01 | 1 (100,000)  | O | 1 (64) relu  |  X  | 1 (1) sigmoid   | 83.43% | 82.61%  |
# | ANN_8 | 0.1 | 1 (100,000)  | O | 1 (64) relu  |  X  | 1 (1) sigmoid   | 77.48% | 72.94%  |
# | ANN_9 | 0.001 | 1 (100,000)  | O | 1 (128) relu  |  X  | 1 (1) sigmoid   | 83.54% | 82.84%  |

# Except for ANN_8 (with a learning rate of 0.1), the model performance only varies in the decimal place, and the best model is ANN_9 (with one hidden layer of 128 nodes) at 82.84% validation accuracy.

# As a result, in this particular case, neural network models failed to outperform logistic regression. This might be due to the high dimensionality and sparse characteristics of the textual data. I have also found a research paper, which compared model performance with high dimension data. According to "An Empirical Evaluation of Supervised Learning in High Dimensions" by Caruana et al.(2008), logistic regression showed as good performance as neural networks, in some cases outperforms neural networks. http://icml2008.cs.helsinki.fi/papers/632.pdf

# Through all the trials above I learned some valuable lessons. Implementing and tuning neural networks is a highly iterative process and includes many trials and errors. Even though the neural network is a more complex version of logistic regression, it doesn't always outperform logistic regression, and sometimes with high dimension sparse data, logistic regression can deliver good performance with much less computation time than neural network.

# In the next post, I will implement a neural network with Doc2Vec vectors I got from the previous post. Hopefully, with dense vectors such as Doc2Vec, a neural network might show some boost. Fingers crossed.

# In[ ]:




