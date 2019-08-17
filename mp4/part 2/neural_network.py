import numpy as np
import time

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):
    start = time.time()
    batch_size = 200
    losses = np.zeros(epoch)
    for e in range(epoch):
        if shuffle:
            idx = np.arange(y_train.size)
            np.random.shuffle(idx)
            x_train = x_train[idx,:]
            y_train = y_train[idx]
        for i in range(y_train.size//batch_size):
            X = x_train[i*batch_size:(i+1)*batch_size,:]
            y = y_train[i*batch_size:(i+1)*batch_size]
            loss, X, w1, w2, w3, w4, b1, b2, b3, b4 = four_nn(X, w1, w2, w3, w4, b1, b2, b3, b4, y)
            losses[e] += loss
        print("epoch: ", e)
        print("loss: ", losses[e])
    end = time.time()
    print("TIME ELAPSED: ", end-start)
    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    total_class = [0]*num_classes

    c = four_nn(x_test, w1, w2, w3, w4, b1, b2, b3, b4, y_test, True)
    for real,guess in zip(y_test, c):
        if real == guess:
            avg_class_rate += 1
            class_rate_per_class[real] += 1
        total_class[real] += 1

    class_rate_per_class = np.divide(class_rate_per_class, total_class)
    avg_class_rate = avg_class_rate / len(y_test)

    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(X, w1, w2, w3, w4, b1, b2, b3, b4, y, test=False):
    Z1, acache1 = affine_forward(X, w1, b1)
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1, w2, b2)
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2, w3, b3)
    A3, rcache3 = relu_forward(Z3)
    F, acache4 = affine_forward(A3, w4, b4)
    if test:
        classifications = []
        for i in range(F.shape[0]):
            classifications.append(np.argmax(F[i]))
        return classifications
    loss, dF = cross_entropy(F,y)
    dA3, dW4, db4 = affine_backward(dF, acache4)
    dZ3 = relu_backward(dA3, rcache3)
    dA2, dW3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)
    dA1, dW2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1, rcache1)
    dX, dW1, db1 = affine_backward(dZ1, acache1)
    # update parameters with gradient Descent
    w1 = w1 - 0.1*dW1
    w2 = w2 - 0.1*dW2
    w3 = w3 - 0.1*dW3
    w4 = w4 - 0.1*dW4
    b1 = b1 - 0.1*db1
    b2 = b2 - 0.1*db2
    b3 = b3 - 0.1*db3
    b4 = b4 - 0.1*db4
    X = X - 0.1*dX
    return loss, X, w1, w2, w3, w4, b1, b2, b3, b4

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    Z = np.add(np.matmul(A, W), b)
    cache = (A, W, b)
    return Z, cache

def affine_backward(dZ, cache):
    A = cache[0]
    W = cache[1]
    dA = np.matmul(dZ, np.transpose(W))
    dW = np.matmul(np.transpose(A), dZ)
    db = np.sum(dZ, axis=0)
    return dA, dW, db

def relu_forward(Z):
    A = np.where(Z < 0, 0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    dZ = np.where(cache < 0, 0, dA)
    return dZ

def cross_entropy(F, y):
    L1 = 0
    L = 0
    for i in range(y.size):
        L1 = F[i][int(y[i])]
        L2 = np.sum(np.exp(F[i]))
        L += L1 - np.log(L2)
    loss = (-1/y.size)*(L)

    one = np.zeros((F.shape[0], F.shape[1]))
    dF = np.zeros((F.shape[0], F.shape[1]))
    for i in range(0, F.shape[0]):
        one[i][int(y[i])] = 1
        dF[i] = one[i] - np.exp(F[i]) / np.sum(np.exp(F[i,:]))
    dF = (-1/y.size)*(dF)
    return loss, dF
