from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    dim = X.shape[1]
    classes = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_score = scores[y[i]] 
        c = np.max(scores)
        new = scores - c
        soft = np.exp(new[y[i]])/np.sum(np.exp(new))
        loss -= np.log(soft)
        dsoft = -1/soft
        dsyi = soft*(1-soft)*dsoft
        dsj = (-1)*soft/np.sum(np.exp(new))*np.exp(new)*dsoft
        dsj[y[i]] = dsyi
        dW += X[i].reshape((dim,1)).dot(dsj.reshape((1,classes)))
    loss /=num_train
    loss += reg*np.sum(W**2)
    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    dim = X.shape[1]
    classes = W.shape[1]
    scores = X.dot(W)
    correct_scores = scores[np.arange(num_train),y]
    c = np.max(scores,axis=1).reshape((num_train,1))
    new = scores - c
    new_correct = new[np.arange(num_train),y]
    soft = (np.exp(new_correct)/np.sum(np.exp(new),axis=1)).reshape((num_train,1))
    loss -= np.sum(np.log(soft))/num_train
    loss += reg*np.sum(W**2)
    dsoft = -1/soft
    dsyi = soft*(1-soft)*dsoft
    dsj = (-1)*soft*dsoft*np.exp(new)/(np.sum(np.exp(new),axis=1).reshape((num_train,1)))
    dsj[np.arange(num_train),y] = dsyi.reshape((num_train))
    dW += X.T.dot(dsj)/num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
