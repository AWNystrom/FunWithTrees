from scipy.stats.mstats import mquantiles
from numpy import arange, median, mean
from code import interact
from sklearn.cluster import KMeans
from sklearn.base import clone
from copy import deepcopy
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

class Node(object):
  def __init__(self, X, Y, max_depth, depth, side):
    self.depth = depth
    self.max_depth = max_depth
    self.terminal = False
    self.side = side
    
    N = len(Y)
    
    if depth < max_depth and N > 1:
      #Figure out where to split
      best_impurity, best_j, best_t, best_mask = None, None, None, None
      n_right, n_left = None, None
      best_clusterer = None
      
      for j in xrange(X.shape[1]):
        x = X[:,j]
        
        """
        clusterer = KMeans(2)
        x = x.reshape(x.shape[0], 1)
        assignments = clusterer.fit_predict(x)
        left_mask = assignments == 0
        Y_left = Y[left_mask]
        Y_right = Y[~left_mask]
        H_left = ((Y_left-Y_left.mean())**2).sum()/len(Y_left)
        H_right = ((Y_right-Y_right.mean())**2).sum()/len(Y_right)
        
        impurity = 0.
          
        if len(Y_left) and len(Y_right):
          impurity += 1.*len(Y_left)/N*H_left + 1.*len(Y_right)/N*H_right
        else:
          impurity = None
          
        if impurity is not None and (impurity < best_impurity or best_impurity is None):
          best_impurity = impurity
          best_j = j
          best_mask = left_mask
          n_right, n_left = len(Y_left), len(Y_right)
          best_clusterer = deepcopy(clusterer)"""
        
        t = median(x)
        #If we split (X,Y) according to j,t, how well would it do by guessing the mean of each group?
        left_mask = x < t
        Y_left = Y[left_mask]
        Y_right = Y[~left_mask]
        
        H_left = (abs(Y_left-Y_left.mean())).sum()/len(Y_left)
        H_right = (abs(Y_right-Y_right.mean())).sum()/len(Y_right)
        impurity = 0.
        
        if len(Y_left) and len(Y_right):
          impurity += 1.*len(Y_left)/N*H_left + 1.*len(Y_right)/N*H_right
        else:
          impurity = None
          
        if impurity is not None and (impurity < best_impurity or best_impurity is None):
          best_impurity = impurity
          best_j = j
          best_t = t
          best_mask = left_mask
          n_right, n_left = len(Y_left), len(Y_right)
      
      self.split_attr = best_j
      self.split_t = best_t
      self.N = N
      #self.clusterer = best_clusterer
      
      if n_left and n_right:
        if n_left:
          self.left = Node(X[best_mask], Y[best_mask], max_depth, depth+1, 'left')
        if n_right:
          self.right = Node(X[~best_mask], Y[~best_mask], max_depth, depth+1, 'right')
      else:
        self.mean = Y.mean()
        self.terminal = True
        self.error = (abs(Y-Y.mean())).sum() / Y.shape[0]
      
    else:
      #This is a terminal node. Fill in the leaves
      self.mean = Y.mean()
      self.terminal = True
      self.error = (abs(Y-Y.mean())).sum() / Y.shape[0]
  
  def __repr__(self):
    if not self.terminal:
      args = (self.depth, self.side, self.terminal, self.split_attr, self.split_t)
      return '''Depth: %s\nSide: %s\nTerminal: %s\nSplit attr: %s\nSplit thresh: %s''' % args
    else:
      args = (self.depth, self.side, self.terminal, self.mean)
      return '''Depth: %s\nSide: %s\nTerminal: %s\nMean: %s''' % args

class Regressor(object):
  def __init__(self, max_depth):
    self.max_depth = max_depth
  
  def fit(self, X, Y):
    self.root = Node(X, Y, self.max_depth, 1, 'root')
  
  def predict_one(self, x):
    cur = self.root
    while True:
      if cur.terminal:
        return cur.mean
      j = cur.split_attr
      t = cur.split_t
      if x[j] < t:
        cur = cur.left
      else:
        cur = cur.right
  
  def get_leaf(self, x):
    cur = self.root
    while True:
      if cur.terminal:
        return cur
      j = cur.split_attr
      t = cur.split_t
      if x[j] < t:
        cur = cur.left
      else:
        cur = cur.right
      
  def predict(self, X):
    cur = self.root
    Y = []
    for x in X:
      Y.append(self.predict_one(x))
    return Y
  
  def __repr__(self):
    ret = ''
    Q = [self.root]
    while Q:
      n = Q.pop(0)
      if not n.terminal:
        Q.append(n.left)
        Q.append(n.right)
      ret += '-'*80
      ret += '\n'
      ret += n.__repr__() + '\n'
    return ret

class Forest(object):
  def __init__(self, ntrees, max_depth, instances_perc, feature_perc):
    self.trees = [Regressor(max_depth) for _ in xrange(ntrees)]
    self.instances_perc = instances_perc
    self.feature_perc = feature_perc
  
  def fit(self, X, Y):
    self.masks = []
    self.fmasks = []
    self.instances_per_tree = int(round(X.shape[0] * self.instances_perc))
    self.features_per_tree = int(round(X.shape[1] * self.feature_perc))
    inds = np.arange(X.shape[0])
    f_inds = np.arange(X.shape[1])
    for t in self.trees:
      mask = np.random.choice(inds, size=self.instances_per_tree, replace=True)
      f_mask = np.random.choice(f_inds, size=self.features_per_tree, replace=False)
      t.fit(X[mask][:, f_mask], Y[mask])
      self.masks.append(mask)
      self.fmasks.append(f_mask)
      
    
  def predict(self, X):
    Y = []
    trees = self.trees
    fmasks = self.fmasks
    for x in X:
      nodes = [t.get_leaf(x[f_mask]) for t, f_mask in zip(trees, fmasks)]
      preds, errors = zip(*[(n.mean, n.error) for n in nodes])
      total_error = sum(errors)
      
      if not total_error:
        Y.append(sum(preds)/len(preds))
      else:
        #Weight each predictor by how well its leaf did.
        norm_errors = [e/total_error for e in errors]
        scores = [1-e for e in norm_errors]
        sum_scores = sum(scores)
        norm_scores = [s/sum_scores for s in scores]
        Y.append(sum(m*s for m,s in zip(preds, norm_scores)))
    return Y

"""class ExtraForest(object):
  def __init__(self, ntrees, max_depth, percent_j_per_tree):
    self.trees = [Regressor(max_depth) for _ in xrange(ntrees)]
  
  def fit(self, X, Y):
    self.masks = []
    for t in self.trees:
      np.random.sample()
      t.fit(X, Y)
    
  def predict(self, X):
    Y = []
    trees = self.trees
    for x in X:
      Y.append(median([t.predict_one(x) for t in trees]))
    return Y"""
      
if __name__ == '__main__':
  from sklearn.metrics import r2_score
  from sklearn.datasets import make_friedman1
  from sklearn.tree import DecisionTreeRegressor
  
  X, Y = make_friedman1(10000, 100)
  X_train, Y_train = X[:9000], Y[:9000]
  X_test, Y_test = X[9000:], Y[9000:]
  
  clf = Forest(50, 10, .7, 1)#Regressor(10)
  clf2 = RandomForestRegressor(50, max_depth=10)
  
  clf.fit(X_train, Y_train)
  clf2.fit(X_train, Y_train)
  
  pred = clf.predict(X_test)
  pred2 = clf2.predict(X_test)
  
  print r2_score(Y_test, pred)
  print r2_score(Y_test, pred2)