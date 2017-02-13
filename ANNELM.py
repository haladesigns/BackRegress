import matplotlib.pyplot as plt
import numpy as np
import random
import time
start_time = time.time()
time_taken_collection = []
test_error_collection = []
total_time_taken = 0
logistic_regression = False


# Sigmoid function and it's derivative

def sigmoid(x):
#     x = np.clip(x, -16, 16)
    return 1.0 / (1.0 + np.exp(-x))
def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# In[1]:

import cPickle, gzip

#Loading MNIST Dataset into training, validation and test set
#traindata 50000 examples, valdata 10000 examples, testdata 10000 examples

data = gzip.open('mnist.pkl.gz', 'rb')
traindata, valdata, testdata = cPickle.load(data)
data.close()


# In[3]:

# On Yann Lecunn's MNIST dataset test errors' web page, he has proposed validation set to have 5000 examples but here the
# 'valdata' has 10000 examples, so I have combined them into 'alldata' and shuffled them finally put them in 'train' and 'val'
# variables with 55000 and 5000 examples respectively. This code cell also converts labels into binary vectors (one-hot encoding)

no_classes = 10
onehoty = np.zeros((traindata[1].shape[0]+valdata[1].shape[0], no_classes)) # np.zeros((55000 + 5000), 10)
for i, y in enumerate(traindata[1]):  #for converting labels into binary vectors e.g 3 to [0 0 0 1 0 0 0 0 0 0]
    z = np.zeros((no_classes))
    z[y] = 1;
    onehoty[i, :] = z
for i, y in enumerate(valdata[1]):   #for converting labels into binary vectors e.g 3 to [0 0 0 1 0 0 0 0 0 0]
    z = np.zeros((no_classes))
    z[y] = 1;
    onehoty[i+traindata[1].shape[0], :] = z
alldata = zip(np.concatenate((traindata[0], valdata[0]), axis=0), onehoty)
np.random.shuffle(alldata)

npxdata = np.zeros((60000, 784)) #images
npydata = np.zeros((60000, 10)) #labels
c = 0
for x, y in alldata:
    npxdata[c, :] = x;
    npydata[c, :] = y
    c += 1

train = zip(npxdata[:55000, :], npydata[:55000, :]) # zip((index 0 - index 55000), (index 55000 - last index))
val = zip(npxdata[55000:, :], npydata[55000:, :])   # zip((index 0 - index 55000), (index 55000 - last index))


# In[4]:

#Converts labels of testdata into binary vectors and rezips them in 'test'

no_classes = 10
onehoty = np.zeros((testdata[1].shape[0], no_classes))
for i, y in enumerate(testdata[1]):   #for converting labels into binary vectors e.g 3 to [0 0 0 1 0 0 0 0 0 0]
    z = np.zeros((no_classes))
    z[y] = 1;
    onehoty[i, :] = z
test = zip(testdata[0], onehoty)


# In[76]:

# Neural Network Class

class Neural_Network(object):
    # Constructor
    def __init__(self, no_neurons=[784, 40, 10], regression=False):
        # It is a 3 elements list where first element is no. of neurons in input layer, second in hidden and third in output
        # layer
        self.no_neurons = no_neurons
        self.regression = regression
        self.num_epochs = 0

        # The naming scheme indicates the type of weight e.g. self.W_ih are weights from 'i'nput to 'h'idden layer
        # Similarly W_ho are weights from 'h'idden to 'o'utput layer, b_h is bias for 'h'idden layer and b_o is bias for
        # 'o'utput layer. The below variables follow the same scheme.
        self.W_ih = np.random.randn(self.no_neurons[1], self.no_neurons[0]) # (40 rows/elements, each with 784 columns)
        self.W_ho = np.random.randn(self.no_neurons[2], self.no_neurons[1]) # (10 rows/elements, each with 40 columns)
        self.b_h = np.random.randn(no_neurons[1], 1) # (40 rows/elements each with 1 column)
        self.b_o = np.random.randn(no_neurons[2], 1) # (10 rows/elements each with 1 column)

        # Contains dot product of W_ih and input + b_h
        self.dproduct_h = np.random.randn(no_neurons[1], 1) # (40 rows/elements  each with 1 column)
        # Contains dot product of W_ho and hidden layer + b_o
        self.dproduct_o = np.random.randn(no_neurons[2], 1) # (10 rows/elements  each with 1 column)

        # Activation Function applied on dproduct, self.a_i is just name for input layer i.e. input image
        self.a_i = np.random.randn(no_neurons[0], 1)
        self.a_h = np.random.randn(no_neurons[1], 1)
        self.a_o = np.random.randn(no_neurons[2], 1)

        # Contains gradients (derivatives) of respective weights and biases
        self.gradW_ih = np.random.randn(self.no_neurons[1], self.no_neurons[0]) 
        self.gradW_ho = np.random.randn(self.no_neurons[2], self.no_neurons[1]) 
        self.gradb_h = np.random.randn(self.no_neurons[1], 1)
        self.gradb_o = np.random.randn(self.no_neurons[2], 1)
        # Contains gradients of loss function
        self.gradP = np.random.randn(self.no_neurons[2], 1)
        
    # Decides which gradient descent to run based on 'mini_batch_size'
    # mini_batch_size: batch size for mini-batch gradient descent
    # no_epochs: no. of iterations over whole training data
    # alpha: learning rate
    # regression: can be turned on/off
    def train(self, training_data, mini_batch_size=1, no_epochs=1, alpha=1.0):
        self.num_epochs = no_epochs
        if mini_batch_size == 1:
            self.stochastic_gradient_descent(training_data, no_epochs, alpha)
        else:
            self.mini_batch_gradient_descent(training_data, mini_batch_size, no_epochs, alpha)

    # predicts class of the input image
    def predict(self, x):
        self.feedforward(x)
        return np.argmax(self.a_o)

    # forward pass through all layers to calculate activation function on the outermost layer (self.a_o)
    def feedforward(self, x):
        self.a_i = x        
        
        # Forward pass through input and hidden layer
        self.dproduct_h = np.dot(self.W_ih, self.a_i) + self.b_h  # net
        self.a_h = sigmoid(self.dproduct_h)  # out (logistic function)
        
        # Forward pass through hidden and output layer
        self.dproduct_o = np.dot(self.W_ho, self.a_h) + self.b_o
        if self.regression == True:
            self.a_o = sigmoid(self.dproduct_o)
            #self.a_o = self.dproduct_o
        else:
            self.a_o = sigmoid(self.dproduct_o)
        
        return self.a_o
    
    # predicts probabilities of different classes for an input image
    def hypothesis(self, x):
        self.a_i = x        
        self.dproduct_h = np.dot(self.W_ih, self.a_i) + self.b_h
        self.a_h = sigmoid(self.dproduct_h)
        
        self.dproduct_o = np.dot(self.W_ho, self.a_h) + self.b_o
        self.a_o = self.dproduct_o
        
        expscores = np.exp(self.a_o)
        probabilities = expscores / np.sum(expscores)
        
        return probabilities
        
    # reverse pass throught all the layers to calculate the gradients w.r.t all the variables (weights and biases)
    def back_propagation(self, x, y):
        self.feedforward(x)
        
        expscores = np.exp(self.a_o)
        probabilities = expscores / np.sum(expscores)

         #--------------------------EDIT---------------------------------
        if self.regression == True:
            #j = 1./m * (-y' * log(sigmoid(X * theta)) - (1 - y))
            #grad = 1./m * X' * (sigmoid(X * theta) - y)

            # The gradient of the logistic regression loss
            self.gradP = probabilities
            self.gradP[np.where(y==1)[0][0]] -= 1

            # Backward Propagation through the output and hidden layer
            self.gradb_o = self.gradP
            self.gradW_ho = np.dot(self.gradP, self.a_h.T)      ##10x1, 1x40
            error_h = (np.dot(self.W_ho.T, self.gradP)) * (derivative_sigmoid(self.dproduct_h))  ## 40x10, 10x1

            #self.gradW_ho = 1. / 10 * (-y * np.log(self.a_o) * self.gradP - (1 - y))
            #self.gradW_ho = 1. / 10 * self.a_h.T * (self.a_o - y)


            #error_h = 1./10 * np.sum( -y * np.log( self.a_o ) - ( 1 - y ) * np.log ( 1 - self.a_o ))
            #error_h = 1./10 * self.a_o * ( self.W_ho - y)

            ###########################################################
            # Backward Propagation through the hidden and input layer #

            ##error_ih = (np.dot(self.W_ho.T, self.gradP)) * (derivative_sigmoid(self.dproduct_h))  ## 40x10, 10x1                                                          #
            #self.gradb_h = error_h                                    #
            ##self.gradb_h = (self.a_i - y)**2 /2
            #self.gradW_ih = np.dot(error_h, self.a_i.T)               #
            ###########################################################

        else:
         #--------------------------EDIT---------------------------------

            # Backward Propagation through the output and hidden layer
            error = (self.a_o - y) * (derivative_sigmoid(self.dproduct_o))
            self.gradb_o = error
            self.gradW_ho = np.dot(error, self.a_h.T)  ##10x1, 1x40
            error_h = (np.dot(self.W_ho.T, error)) * (derivative_sigmoid(self.dproduct_h))  ## 40x10, 10x1

        # Backward Propagation through the hidden and input layer
        self.gradb_h = error_h
        self.gradW_ih = np.dot(error_h, self.a_i.T)

    def stochastic_gradient_descent(self, training_data, no_epochs, alpha):
        tot_epochs = no_epochs
        while no_epochs > 0:
            # Randomly shuffle data after each epoch
            random.shuffle(training_data)
            #random.shuffle(training_data)
            
            count = 0;
            for x, y in training_data:
                count += 1
                
                #Adding one dimension for dot product
                x = x[:, np.newaxis];
                y = y[:, np.newaxis];
                
                # Calculates the graidents w.r.t weights and biases, so that we can find the direction in which
                # loss will reduce
                self.back_propagation(x, y)
                
                # Updating the weights using the gradient and the learning rate
                self.W_ih = self.W_ih - alpha * self.gradW_ih;
                self.b_h  = self.b_h  - alpha * self.gradb_h;

                if self.regression == True:
                    #theta = theta - (alpha / m) * X' * (1./(1 + exp(-X * theta)) -y)
                    #self.W_ho = self.W_ho - (alpha/10) * self.a_o * (1./(1 + np.exp(-self.a_o * self.W_ho)) - y)
                    #self.W_ho = self.W_ho - (alpha/10) * self.a_o * (self.a_o * self.W_ho- y)

                    #if no_epochs == 5:
                        # theta = theta - (alpha / m) * X' * (1./(1 + exp(-X * theta)) -y)

                        # self.W_ho = self.W_ho - (alpha / 10) * self.a_o * (self.a_o * self.W_ho- y)
                    #self.W_ho = self.W_ho - alpha * self.gradW_ho;
                    #self.b_o = self.b_o - alpha * self.gradb_o;
                        # self.b_o = self.b_o - (alpha / 10) * self.a_o * (self.a_o * self.b_o - y)

                    #self.W_ho = self.W_ho - ((alpha / 10)) * self.a_h * (1. / (1 + np.exp(-self.a_h * self.W_ho)) - y)
                    self.W_ho = self.W_ho - (alpha / 10) * self.a_o * (1. / (1 + np.exp(-self.a_o * self.W_ho)) - y)
                    self.b_o = self.b_o - (alpha / 10) * self.a_o * (1. / (1 + np.exp(-self.a_o * self.b_o)) - y)
                else:
                    self.W_ho = self.W_ho - alpha * self.gradW_ho;
                    self.b_o  = self.b_o  - alpha * self.gradb_o;
                
                # Show loss after every 2000th iteration
                if count % 2000 == 0:
                    csum = 0
                    l = 0
                    for x, y in val:
                        x = x[:, np.newaxis];
                        y = y[:, np.newaxis];
                        if self.regression == True:
                            feed = self.hypothesis(x)
                            #feed = self.feedforward(x)
                        else:
                            feed = self.feedforward(x)
                        #csum += np.sum(np.abs(feed-y))   ##L1
                        #print "Validation L1 Loss: {}".format((1.0/len(val))*(csum))

                        csum += np.sum(np.square(feed-y))  ##L2

                    #  --------------------------EDIT---------------------------------
                    #     l += -np.log(feed[np.where(y==1)[0][0], 0])
                    #
                    # if self.regression == True:
                    #     print "Validation Log Loss: {}, L2 Loss: {}".format((1.0/len(val))*l, (1.0/len(val))*np.sqrt(csum))
                    # else:
                    #  --------------------------EDIT---------------------------------

                    #print "Validation L2 Loss: {}".format((1.0/len(val))*np.sqrt(csum))
                    
#             print count
            no_epochs -= 1
            print "Epochs Completed: {}".format(tot_epochs-no_epochs)
            print "Y: ", y.T
            print
            print "OUTPUT: ", self.a_o.T
            print "Prediction: ", np.argmax(self.a_o.T)
            print

        total = len(valdata[0])
        count = 0
        for num in range(0, total):
            e = valdata[0][num]
            e = e[:, np.newaxis]

            if nn.predict(e) == valdata[1][num]:
                count += 1
            num += 1
        success_rate = (count * 100.0) / total
        print "Number of hidden neurons: ", nn.no_neurons[1]
        print "Count: ", count
        print "Total: ", total
        print "Prediction Success rate : {0:.2f}%".format(success_rate)
        print "********************************************************"
        print

    # --------------------------EDIT---------------------------------
    # def mini_batch_gradient_descent(self, training_data, mini_batch_size,
    #                                 no_epochs, alpha):
    #     tot_epochs = no_epochs;
    #     while no_epochs > 0:
    #         # Randomly shuffle data after each epoch
    #         random.shuffle(training_data)
    #
    #         # Create batches of training data of size = 'mini_batch_size'
    #         batches = [];
    #         for i in range(0, len(training_data), mini_batch_size):
    #             batches.append(training_data[i:i + mini_batch_size])
    #
    #         for bn, mini_batch in enumerate(batches):
    #             # Since we are finding gradient over a mini-batch, we will have to sum gradient over all the examples in
    #             # the mini-batch, so this is the accumulator for the gradients (follows same naming scheme)
    #             acc_gradW_ih = np.zeros((self.no_neurons[1], self.no_neurons[0]))
    #             acc_gradW_ho = np.zeros((self.no_neurons[2], self.no_neurons[1]))
    #             acc_gradb_h = np.zeros((self.no_neurons[1], 1))
    #             acc_gradb_o = np.zeros((self.no_neurons[2], 1))
    #
    #             for x, y in mini_batch:
    #                 # Adding one dimension for dot product
    #                 x = x[:, np.newaxis];
    #                 y = y[:, np.newaxis];
    #
    #                 # Calculates the graidents w.r.t weights and biases, so that we can find the direction in which
    #                 # loss will reduce
    #                 self.back_propagation(x, y)
    #
    #                 #Accumulating gradient as stated above
    #                 acc_gradW_ih = acc_gradW_ih + self.gradW_ih
    #                 acc_gradW_ho = acc_gradW_ho + self.gradW_ho
    #                 acc_gradb_h = acc_gradb_h + self.gradb_h
    #                 acc_gradb_o = acc_gradb_o + self.gradb_o
    #
    #             # Updating the weights using the gradient and the learning rate
    #             self.W_ih = self.W_ih - (float(alpha) / mini_batch_size) * acc_gradW_ih;
    #             self.b_h  = self.b_h  - (float(alpha) / mini_batch_size) * acc_gradb_h;
    #
    #             self.W_ho = self.W_ho - (float(alpha) / mini_batch_size) * acc_gradW_ho;
    #             self.b_o  = self.b_o  - (float(alpha) / mini_batch_size) * acc_gradb_o;
    #
    #             # Showing loss after every 2000 examples
    #             if bn % np.floor(2000.0/mini_batch_size) == 0:
    #                 #print "Calculating Validation Loss"
    #                 csum = 0
    #                 l = 0
    #                 for x, y in val:
    #                     x = x[:, np.newaxis];
    #                     y = y[:, np.newaxis];
    #                     if self.regression == True:
    #                         feed = self.hypothesis(x)
    #                     else:
    #                         feed = self.feedforward(x)
    #                     #csum += np.sum(np.abs(feed-y))   ##L1
    #                     #print "Validation L1 Loss: {}".format((1.0/len(val))*(csum))
    #
    #                     csum += np.sum(np.square(feed-y))  ##L2
    #                     l += -np.log(feed[np.where(y==1)[0][0], 0])
    #
    #                 if self.regression == True:
    #                     print "Validation Log Loss: {}, L2 Loss: {}".format((1.0/len(val))*l, (1.0/len(val))*np.sqrt(csum))
    #                 else:
    #                     print "Validation L2 Loss: {}".format((1.0/len(val))*np.sqrt(csum))
    #
    #         no_epochs -= 1
    #         print "Epochs Completed: {}".format(tot_epochs-no_epochs)
    #  --------------------------EDIT---------------------------------

# In[ ]:


# Creating object for neural network
#  nn = Neural_Network([784, 10, 10], regression=False)
#  nn = Neural_Network([784, 20, 10], regression=False)
#  nn = Neural_Network([784, 30, 10], regression=False)
#  nn = Neural_Network([784, 40, 10], regression=False)
## nn = Neural_Network([784, 50, 10], regression=False)
#  nn = Neural_Network([784, 60, 10], regression=False)
#  nn = Neural_Network([784, 70, 10], regression=False)
#  nn = Neural_Network([784, 80, 10], regression=False)
#  nn = Neural_Network([784, 90, 10], regression=False)
#  nn = Neural_Network([784, 100, 10], regression=False)

#  nn = Neural_Network([784, 10, 10], regression=True)
#  nn = Neural_Network([784, 20, 10], regression=True)
#  nn = Neural_Network([784, 30, 10], regression=True)
#  nn = Neural_Network([784, 40, 10], regression=True)
#  nn = Neural_Network([784, 50, 10], regression=True)
#  nn = Neural_Network([784, 60, 10], regression=True)
#  nn = Neural_Network([784, 70, 10], regression=True)
#  nn = Neural_Network([784, 80, 10], regression=True)
#  nn = Neural_Network([784, 90, 10], regression=True)
#  nn = Neural_Network([784, 100, 10], regression=True)

# Training the Neural network over 'train'
##nn.train(train, mini_batch_size=1, no_epochs=10, alpha=0.1)
#nn.train(train, mini_batch_size=8, no_epochs=20, alpha=0.1)
#for hidden_neurons in range(10, 60, 10):
#for hidden_neurons in range(10, 70, 10):
#for hidden_neurons in range(10, 20, 10):
#for hidden_neurons in range(10, 110, 10):
#for hidden_neurons in range(10, 70, 10):
#nn = Neural_Network([784, 10, 10], regression=True)
#for hidden_neurons in range(10, 110, 10):
for hidden_neurons in range(50, 60, 10):
#for hidden_neurons in range(10, 70, 10):
    start_time = time.time()
    nn = Neural_Network([784, hidden_neurons, 10], regression=False)
    #nn = Neural_Network([784, hidden_neurons, 10], regression=False)

    #nn.train(train, mini_batch_size=1, no_epochs=20, alpha=0.1)
    #nn.train(train, mini_batch_size=1, no_epochs=10, alpha=0.1)
    nn.train(train, mini_batch_size=1, no_epochs=5, alpha=0.25)
    #nn.train(train, mini_batch_size=1, no_epochs=1, alpha=0.1)

    time_taken = (time.time() - start_time) / 60
    time_taken_collection.append(time_taken)
    total_time_taken += time_taken
    logistic_regression = nn.regression

    no_err = 0
    for x, y in test:
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        feed = nn.predict(x)
    #     print feed
    #     print np.where(y==1)[0][0]
        if feed != np.where(y==1)[0][0]:
            no_err += 1
    test_error_collection.append(float(100.0*no_err/len(test)))

#objects = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#objects = [10, 20, 30, 40, 50, 60]
#objects = [10, 20, 30, 40]
#objects = [10]
objects = [50]

# In[79]:

# Given example no, shows
#  example_no = 2;
example_no = 5;

ex = valdata[0][example_no]
ex = ex[:, np.newaxis]
print "Example No. {} in validation set".format(example_no)
print "Probabilities of different classes: {}".format(np.squeeze(nn.hypothesis(ex)))
print
print "Predicted Class: {}".format(nn.predict(ex))
print "   Actual Class: {}".format(valdata[1][example_no])
print "-------------------------------------------------------------"

# In[82]:

## FOR EVALUATING FINAL TEST ERROR RATE

#5.55 with 50 neurons in hidden layer

#5.44 with 40 neurons in hidden layer (softmax) lr=0.0001

#4.84, batch size=16, neurons=40

print "Number of errors: {}".format(no_err)
print "Test Error Rate: {} %".format(100.0*no_err/len(test))
print "Time taken: {} mins".format(time_taken)
print "time_taken_collection {}".format(time_taken_collection)



# objects = [10,20]
# objects = [10]
# y_pos = np.arange(len(objects))
# # time_taken_collection = [10,8,6,4,2,1]
#
# plt.bar(y_pos, time_taken_collection, align='center', alpha=0.8, width=0.5, color = "green")
# plt.xticks(y_pos, objects)
# plt.grid(True)
# plt.ylabel('Minutes')
# plt.xlabel('No. of Hidden Neurons')
# plt.title('Time taken')
# plt.text(4, 4, r'10 epochs')
# plt.text(4, 3.5, r'regression = False')
# plt.show()


# #N = 6
# N = 1
# #time_taken_collectionStd = (2, 3, 4, 1, 2, 1)
# time_taken_collectionStd = (2)
#
# ind = np.arange(N)  # the x locations for the groups
# width = 0.35       # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, time_taken_collection, width, color='b', yerr=time_taken_collectionStd)
#
# #test_error_collectionStd = (3, 5, 2, 3, 3, 4)
# test_error_collectionStd = (3)
# rects2 = ax.bar(ind + width, test_error_collection, width, color='y', yerr=test_error_collectionStd)
#
# # add some text for labels, title and axes ticks
# ax.set_ylabel('Minutes')
# ax.set_title('Time taken + Error percentage')
# ax.set_xticks(ind + width)
# #ax.set_xticklabels(('10', '20', '30', '40', '50', '60'))
# ax.set_xticklabels(('10'))
#
# ax.legend((rects1[0], rects2[0]), ('Time', 'Error'))
#
#
# def autolabel(rects):
#     # attach some text labels
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                 '%d' % int(height),
#                 ha='center', va='bottom')
#
# autolabel(rects1)
# autolabel(rects2)
x1 = objects
y1 = time_taken_collection

x2 = objects
y2 = test_error_collection

# plt.subplot(2, 1, 1)
# plt.bar(x1,y1,align='center', alpha=0.8, width=0.8, color = "blue")
# plt.title('Time taken')
# plt.xticks(x1, objects)
# plt.ylabel('Minutes')
# plt.grid()
#
# plt.subplot(2, 1, 2)
# plt.bar(x2,y2,align='center', alpha=0.8, width=0.8, color = "green")
# plt.xlabel('No. of Hidden Neurons')
# plt.ylabel('Test %Error')
#
# plt.grid()

ax1 = plt.subplot(2,1,1)
ax1.set_ylabel('Time (Minutes)')
ax1.set_title('Time taken')
#ax1.axis([0,70, 0,40])
ax1.axis([0,max(x1) + 10, 0,max(y1) + 5])
ax1.text(1, max(y1) + 3, r'5 epochs')
ax1.text(1, max(y1) + 2, r'Regression: {}'.format(logistic_regression))
ax1.text(1, max(y1) + 1, r'Total Time: {0:.2f}'.format(total_time_taken))
ax1.grid()

ax2 = plt.subplot(2,1,2)
ax2.set_ylabel('Test Error Rate(%)')
ax2.set_xlabel('No. of Neurons')
#ax2.axis([0,70, 0,40])
ax2.axis([0,max(x2) + 10, 0,max(y2) + 5])
ax2.text(1, max(y2), r'5 epochs')
ax2.text(1, max(y2) - 3, r'Regression: {}'.format(logistic_regression))
#ax2.text(1, max(y2) - 6, r'Shuffle: True')
ax2.text(1, max(y2) - 6, r'Shuffle: True')
#ax2.text(1, max(y2) - 3, r'Shuffle training_data: True')
ax2.grid()

rects1 = ax1.bar(x1, y1, color='b', align='center')
rects2 = ax2.bar(x2, y2, color='g', align='center')

def autolabel(rects, axes): #attach some text labels
    for rect in rects:
        height = rect.get_height()
        #axes.text(rect.get_x() + rect.get_width()/2.,height,'%d' % int(height),ha='center', va='bottom')
        if axes == ax1:
            axes.text(rect.get_x() + rect.get_width()/2.0,height,'%04.2f' % height,ha='center', va='bottom')
        else:
            axes.text(rect.get_x() + rect.get_width() / 2.0, height, '{0:.2f}%'.format(height), ha='center', va='bottom')

autolabel(rects1, ax1)
autolabel(rects2, ax2)

plt.show()
