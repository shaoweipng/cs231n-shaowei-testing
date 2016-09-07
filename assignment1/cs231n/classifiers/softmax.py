import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]
  #print 'W.shape'
  #print W.shape
  num_train = X.shape[0]
  #print 'X.shape'
  #print X.shape

  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    #print 'scores.shape'
    #print scores.shape
    normalize_scores = np.exp(scores) / np.sum(np.exp(scores))
    loss += - np.log(normalize_scores[y[i]])
    
    #print 'normalize_scores.shape'
    #print normalize_scores.shape
    #print normalize_scores

    #print np.reshape(normalize_scores, (num_classes,1)) * X[i, :]
    
    #print 'X[i,:].shape'
    #print X[i,:].shape
    #print X[i,:]
    
    #print 'np.reshape(normalize_scores, (num_classes,1)).shape'
    #print np.reshape(normalize_scores, (num_classes,1)).shape
    #print np.reshape(normalize_scores, (num_classes,1))
    
    #dscore = X[i, :] * np.reshape(normalize_scores, (num_classes,1)) 
    
    dscore = X[i,:].reshape(3073,1) * normalize_scores
    
    #print 'dscore.shape'
    #print dscore.shape
    #print dscore
    dscore[:,y[i]] -= X[i, :]
    #print dscore[:,y[i]].shape
    dW += dscore

  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  dW = dW /num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #print 'scores.shape'
  #print scores.shape
  #print scores
  #print 'np.max(scores,axis=1)'
  #print np.max(scores,axis=1)
  maxscore = np.max(scores,axis=1)
  #print 'maxscore.shape'
  #print maxscore.shape
  shifted_scores = scores -maxscore[:,np.newaxis]
  #print 'shifted_scores.shape'
  #print shifted_scores.shape
  #print shifted_scores


  probs = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), 1)[:,np.newaxis]
  #print 'probs.shape'
  #print probs.shape
  losses = -np.log( probs[xrange(num_train), y] )

  loss = losses.sum() / num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  true_indices = np.zeros(probs.shape)
  true_indices[xrange(num_train),y] = 1
  dW = - X.transpose().dot(true_indices - probs)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

