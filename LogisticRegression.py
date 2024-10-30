import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def Sigmoid(z):
    #returns the sigmoid function 
    return 1.0/(1+np.exp(-z))

def LogisticRegression():
    x = 10

def mean_squared(y_true, y_predict):
    cost = np.sum((y_true-y_predict)**2) / len(y_true)
    return cost
def gradient_decent():
    x= 0
    
def KFoldCrossValidation(feature_columns, y_values, k = 10):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for i, (training, testing) in enumerate(kf.split(feature_columns)):
        X_training, X_test = feature_columns.iloc[training], feature_columns.iloc[testing]
        y_training, y_test = y_values.iloc[training], y_values.iloc[testing]

    
def main():
    print("Welcome to Logistic Regression")
    df = pd.read_csv('MNIST_CV.csv')
    print(df.info())
    y_values = df['label']
    X_values = df.iloc[:, df.columns != 'label']
    KFoldCrossValidation(X_values, y_values, 10)
    
    logistic = Sigmoid(X_training.dot(coeffient.T)
    costp = (-logistic + np.squeeze(y_training)).T.dot(X_training)
    y_training = np.squeeze(y_training)
    coeffient = coeffient + learning_rate * costp

    #likeilywood function y.log(p(xi)-((1-y)(log(1-p(xi)))))
    lw1 = (y_training *np.log(logistic))
    lw2 = ((1-y_training)*np.log(1 - logistic))
    costv = +lw1 + lw2

    # mean of function
    costf = np.mean(costv)
    cost.append(costf)



if __name__ == "__main__":
    main()