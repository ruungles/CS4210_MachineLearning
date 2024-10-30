import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def LogisticRegression():
    x = 10

def mean_squared(y_true, y_predict):
    cost = np.sum((y_true-y_predict)**2) / len(y_true)
    return cost
def gradient_decent():
    x= 0
    
def KFoldCrossValidation(dataFrame, feature_columns, y_values, k = 5):
    kf = KFold(n_splits= k, shuffle=True, random_state=42)
    
def main():
    print('hello')

if __name__ == "__main__":
    main()