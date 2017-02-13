
# coding: utf-8

# In[2]:

import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(x):
#     x = np.clip(x, -16, 16)
    return 1.0 / (1.0 + np.exp(-x))
def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# In[3]:

import cPickle, gzip

data = gzip.open('mnist.pkl.gz', 'rb')
traindata, valdata, testdata = cPickle.load(data)
data.close()


# In[4]:

no_classes = 10
onehoty = np.zeros((traindata[1].shape[0]+valdata[1].shape[0], no_classes))
for i, y in enumerate(traindata[1]):
    z = np.zeros((no_classes))
    z[y] = 1;
    onehoty[i, :] = z
for i, y in enumerate(valdata[1]):
    z = np.zeros((no_classes))
    z[y] = 1;
    onehoty[i+traindata[1].shape[0], :] = z
alldata = zip(np.concatenate((traindata[0], valdata[0]), axis=0), onehoty)
np.random.shuffle(alldata)


# In[5]:

npxdata = np.zeros((60000, 784))
npydata = np.zeros((60000, 10))
c = 0
for x, y in alldata:
    npxdata[c, :] = x;
    npydata[c, :] = y
    c += 1
#     print x.shape
#     print y.shape


# In[6]:

train = zip(npxdata[:55000, :], npydata[:55000, :])
val = zip(npxdata[55000:, :], npydata[55000:, :])


# In[7]:

no_classes = 10
onehoty = np.zeros((testdata[1].shape[0], no_classes))
for i, y in enumerate(testdata[1]):
    z = np.zeros((no_classes))
    z[y] = 1;
    onehoty[i, :] = z
test = zip(testdata[0], onehoty)


# In[34]:

class Neural_Network(object):
    def __init__(self, no_neurons=[784, 40, 10]):
        self.no_neurons = no_neurons

        self.W_ih = np.random.randn(self.no_neurons[1], self.no_neurons[0]) 
        self.W_ho = np.random.randn(self.no_neurons[2], self.no_neurons[1]) 
        self.b_h = np.random.randn(no_neurons[1], 1)
        self.b_o = np.random.randn(no_neurons[2], 1)

        self.dproduct_h = np.random.randn(no_neurons[1], 1)
        self.dproduct_o = np.random.randn(no_neurons[2], 1)

        self.a_i = np.random.randn(no_neurons[0], 1)
        self.a_h = np.random.randn(no_neurons[1], 1)
        self.a_o = np.random.randn(no_neurons[2], 1)

        self.gradW_ih = np.random.randn(self.no_neurons[1], self.no_neurons[0]) 
        self.gradW_ho = np.random.randn(self.no_neurons[2], self.no_neurons[1]) 
        self.gradb_h = np.random.randn(self.no_neurons[1], 1)
        self.gradb_o = np.random.randn(self.no_neurons[2], 1)
        self.gradP = np.random.randn(self.no_neurons[2], 1)
        
    def train(self, training_data, mini_batch_size=1, no_epochs=1, alpha=1.0):
        if mini_batch_size == 1:
            self.incremental_gradient_descent(training_data, no_epochs, alpha)
        else:
            self.stochastic_gradient_descent(training_data, mini_batch_size, no_epochs, alpha)

    def predict(self, x):
        self.feedforward(x)
        return np.argmax(self.a_o)

    def feedforward(self, x):
        self.a_i = x        
        self.dproduct_h = np.dot(self.W_ih, self.a_i) + self.b_h
#         print "HAH"
#         print self.dproduct_h.shape
        self.a_h = sigmoid(self.dproduct_h)
        
        self.dproduct_o = np.dot(self.W_ho, self.a_h) + self.b_o
#         print self.dproduct_o.shape
        self.a_o = self.dproduct_o
#         print self.a_o.shape
        
        return self.a_o
    
    def hypothesis(self, x):
        self.a_i = x        
        self.dproduct_h = np.dot(self.W_ih, self.a_i) + self.b_h
#         print "HAH"
#         print self.dproduct_h.shape
        self.a_h = sigmoid(self.dproduct_h)
        
        self.dproduct_o = np.dot(self.W_ho, self.a_h) + self.b_o
#         print self.dproduct_o.shape
        self.a_o = self.dproduct_o
#         print self.a_o.shape
        expscores = np.exp(self.a_o)
        probabilities = expscores / np.sum(expscores)
        
        return probabilities        
        
    def back_propagation(self, x, y):
        self.feedforward(x)
        
        expscores = np.exp(self.a_o)
        probabilities = expscores / np.sum(expscores)
#         print probabilities.shape
        
        self.gradP = probabilities
        self.gradP[np.where(y==1)[0][0]] -= 1
        
#         error = (self.a_o - y) * (derivative_sigmoid(self.dproduct_o))
        self.gradb_o = self.gradP
        self.gradW_ho = np.dot(self.gradP, self.a_h.T)      ##10x1, 1x40
        
        error_h = (np.dot(self.W_ho.T, self.gradP)) * (derivative_sigmoid(self.dproduct_h))   ## 40x10, 10x1
#         print (self.dproduct_h).shape
#         print error_h.shape
        self.gradb_h = error_h
        self.gradW_ih = np.dot(error_h, self.a_i.T)

    def incremental_gradient_descent(self, training_data, no_epochs, alpha):
        tot_epochs = no_epochs;
        while no_epochs > 0:
            random.shuffle(training_data)
#             perm = np.random.permutation(len(training_data[1]));
#             xtrain = training_data[0]
#             ytrain = training_data[1]
#             xtrain = xtrain[perm, :]
#             ytrain = ytrain[perm]
            count = 0;
            for x, y in training_data:
#             for i in range(len(training_data[1])):
#                 x = xtrain[i]
#                 y = ytrain[i]
                count += 1
#                 print x, y
#                 print x.shape
                x = x[:, np.newaxis];
#                 print y.shape
                y = y[:, np.newaxis];
                self.back_propagation(x, y)
                
#                 print self.W_ih, self.b_h, self.W_ho, self.b_o
                
                self.W_ih = self.W_ih - alpha * self.gradW_ih;
                self.b_h  = self.b_h  - alpha * self.gradb_h;

                self.W_ho = self.W_ho - alpha * self.gradW_ho;
                self.b_o  = self.b_o  - alpha * self.gradb_o;   
#                 print count
                
                if count % 1000 == 0:
#                     print "Calculating Validation Loss"
                    csum = 0
                    l = 0
                    for x, y in val:
                        x = x[:, np.newaxis];
                        y = y[:, np.newaxis];
                        feed = self.hypothesis(x)
#                         csum += np.sum(np.abs(feed-y))   ##L1
#                     print "Validation L1 Loss: {}".format((1.0/len(val))*(csum))

                        csum += np.sum(np.square(feed-y))  ##L2

                        l += -np.log(feed[np.where(y==1)[0][0], 0])
#                         print feed[np.where(y==1)[0][0], 0], -np.log(feed[np.where(y==1)[0][0], 0])

                    print "Log Loss: {}, L2 Loss: {}".format((1.0/len(val))*l, (1.0/len(val))*np.sqrt(csum))
#                     print "Validation L2 Loss: {}".format((1.0/len(val))*np.sqrt(csum))
                    
#             print count
            no_epochs -= 1
            print "Epochs Completed: {}".format(tot_epochs-no_epochs)
            
    def stochastic_gradient_descent(self, training_data, mini_batch_size,
                                    no_epochs, alpha):
        tot_epochs = no_epochs;
        while no_epochs > 0:
            random.shuffle(training_data)
            batches = [];
            for i in range(0, len(training_data), mini_batch_size):
                batches.append(training_data[i:i + mini_batch_size])
                
            for bn, mini_batch in enumerate(batches):
                acc_gradW_ih = np.zeros((self.no_neurons[1], self.no_neurons[0]))
                acc_gradW_ho = np.zeros((self.no_neurons[2], self.no_neurons[1]))
                acc_gradb_h = np.zeros((self.no_neurons[1], 1))
                acc_gradb_o = np.zeros((self.no_neurons[2], 1))
                for x, y in mini_batch:
                    x = x[:, np.newaxis];
                    y = y[:, np.newaxis];
                    
                    self.back_propagation(x, y)
                    
                    acc_gradW_ih = acc_gradW_ih + self.gradW_ih
                    acc_gradW_ho = acc_gradW_ho + self.gradW_ho
                    acc_gradb_h = acc_gradb_h + self.gradb_h
                    acc_gradb_o = acc_gradb_o + self.gradb_o
                    
                self.W_ih = self.W_ih - (float(alpha) / mini_batch_size) * acc_gradW_ih;
                self.b_h  = self.b_h  - (float(alpha) / mini_batch_size) * acc_gradb_h;

                self.W_ho = self.W_ho - (float(alpha) / mini_batch_size) * acc_gradW_ho;
                self.b_o  = self.b_o  - (float(alpha) / mini_batch_size) * acc_gradb_o;

#                 if count % 1000 == 0:
                if bn % 100 == 0:
#                     print "Calculating Validation Loss"
                    csum = 0
                    l = 0
                    for x, y in val:
                        x = x[:, np.newaxis];
                        y = y[:, np.newaxis];
                        feed = self.hypothesis(x)
#                         csum += np.sum(np.abs(feed-y))   ##L1
#                     print "Validation L1 Loss: {}".format((1.0/len(val))*(csum))

                        csum += np.sum(np.square(feed-y))  ##L2

                        l += -np.log(feed[np.where(y==1)[0][0], 0])
#                         print feed[np.where(y==1)[0][0], 0], -np.log(feed[np.where(y==1)[0][0], 0])

                    print "Log Loss: {}, L2 Loss: {}".format((1.0/len(val))*l, (1.0/len(val))*np.sqrt(csum))
#                     print "Validation L2 Loss: {}".format((1.0/len(val))*np.sqrt(csum))
                    
            no_epochs -= 1
            print "Epochs Completed: {}".format(tot_epochs-no_epochs)


# In[74]:

nn = Neural_Network([784, 300, 10])


# In[75]:

nn.train(train, mini_batch_size=16, no_epochs=30, alpha=0.1)


# In[76]:

## FOR EVALUATING FINAL TEST ERROR RATE

#5.55 with 50 neurons in hidden layer

#5.44 with 40 neurons in hidden layer (softmax) lr=0.0001

#4.84, batch size=16, neurons=40

no_err = 0;
for x, y in test:
    x = x[:, np.newaxis];
    y = y[:, np.newaxis];
    feed = nn.predict(x)
#     print feed
#     print np.where(y==1)[0][0]
    if feed != np.where(y==1)[0][0]:
        no_err += 1
print no_err
print "Test Error Rate: {} %".format(100.0*no_err/len(test))    


# In[67]:

ex = valdata[0][2]
ex = ex[:, np.newaxis]
print nn.predict(ex)


# In[68]:

valdata[1][2]


# In[69]:

ex = valdata[0][2]
ex = ex[:, np.newaxis]
print nn.hypothesis(ex)

