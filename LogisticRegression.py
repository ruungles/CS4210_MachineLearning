import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import linear_model

class LogitRegression():
    def __init__ (self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
    def fit(self, X, y):
        self.training, self.features = X.shape
        self.weight = np.zeros(self.features)
        self.bias = 0
        self.X = X
        self.y = y 

        for _ in range(self.iterations):
            self.update_weights(X, y)
        return self


    def update_weights (self,X,y):
        sigmoid = 1.0 / (1.0 + np.exp(-(X.dot(self.weight) + self.bias) ) )

        # gradients
        tmp = (sigmoid - self.y.T)
        tmp = np.reshape(tmp, self.training)
        dw = np.dot(self.X.T, tmp) / self.training
        db = np.sum(tmp) / self.training

        self.weight = self.weight - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db

        return self 

    def predict (self, X) :
        proba = self.predict_proba(X)
        return np.where(proba > 0.5, 1, 0)

    def predict_probability(self, X):
        return 1 / ( 1 + np.exp( - ( X.dot( self.weight ) + self.bias ) ) ) 

def main():
    print("Welcome to Logistic Regression")
    file_name = "\\MNIST_CV.csv"
    df = pd.read_csv(file_name)
    
    #Split into Features(X) and Labels(y)
    X_values = df.iloc[:, df.columns != 'label'].values
    y_values = df['label'].values
    y_values = np.where(df['label'].values == 6, 0, 1)

    print(X_values)
    print(y_values)
    
    kfold = KFold(n_splits=10, shuffle =True, random_state=42)
    tpr_list1 = []
    fpr_list1 = []
    
    '''
    tpr_list2 = []
    fpr_list2 = []
    '''
    #Using Scikit-learn's KFold-Cross Validation 
    for fold, (training_index, testing_index) in enumerate(kfold.split(X_values), 1):
        X_train, X_test = X_values[training_index], X_values[testing_index]
        Y_train, Y_test = y_values[training_index], y_values[testing_index]
        #Testing my model
        model1 = LogitRegression(learning_rate= 0.000001, iterations= 100)
        model1.fit(X_train, Y_train)
        Y_probability1 = model1.predict_probability(X_test)

        fpr1, tpr1, _ = roc_curve(Y_test, Y_probability1)
        tpr_list1.append(np.interp(np.linspace(0, 1, 100), fpr1, tpr1))
        fpr_list1.append(np.linspace(0, 1, 100))
        
        '''
        model2 = linear_model.LogisticRegression()
        model2.fit(X_train, Y_train)
        Y_probability2 = model2.predict(X_test)
        
        fpr2, tpr2, _ = roc_curve(Y_test, Y_probability2)
        tpr_list2.append(np.interp(np.linspace(0, 1, 100), fpr2, tpr2))
        fpr_list2.append(np.linspace(0, 1, 100))
        '''

    tpr_dataFrame1 = pd.DataFrame(tpr_list1)
    print(tpr_dataFrame1)
    fpr_dataFrame1 = pd.DataFrame(fpr_list1)
    print(fpr_dataFrame1)
    
    mean_tpr1 = np.mean(tpr_list1, axis=0)
    mean_fpr1 = np.mean(fpr_list1, axis=0)
    plt.plot(mean_fpr1, mean_tpr1, label=f"ROC Curve (AUC = {np.trapz(mean_tpr1, mean_fpr1):.2f})")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Average ROC Curve from 10-Fold Cross-Validation")
    plt.legend()
    plt.show()

    '''
    tpr_dataFrame2 = pd.DataFrame(tpr_list2)
    print(tpr_dataFrame2)
    fpr_dataFrame2 = pd.DataFrame(fpr_list2)

    mean_tpr2 = np.mean(tpr_list2, axis=0)
    
    mean_fpr2 = np.mean(fpr_list2, axis=0)
    plt.plot(mean_fpr2, mean_tpr2, label=f"ROC Curve (AUC = {np.trapz(mean_tpr2, mean_fpr2):.2f})")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Average ROC Curve from 10-Fold Cross-Validation")
    plt.legend()
    plt.show()
    '''

if __name__ == "__main__":
    main()