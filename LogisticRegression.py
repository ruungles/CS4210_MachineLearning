import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

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
        sigmoid = 1.0 / (1.0 + np.exp(-(self.X.dot(self.weight) + self.bias) ) )

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
    file_name = "C:\\Users\\rlars\\OneDrive\\Desktop\\MachineLearning\\Assignments\\HW4\\CS4210_MachineLearning\\DATA.csv"
    df = pd.read_csv(file_name)
    
    #Split into Features(X) and Labels(y)
    X_values = df.iloc[:, df.columns != 'label'].values
    y_values = df['label'].values
    y_values = np.where(df['label'].values == 6, 0, 1)

    print(X_values)
    print(y_values)

    scaler = StandardScaler()
    X_values = scaler.fit_transform(X_values)

    kfold = KFold(n_splits=10, shuffle =True, random_state=42)
    tpr_list = []
    fpr_list = []

    for fold, (training_index, testing_index) in enumerate(kfold.split(X_values), 1):
        X_train, X_test = X_values[training_index], X_values[testing_index]
        Y_train, Y_test = y_values[training_index], y_values[testing_index]
        model = LogitRegression(learning_rate= 0.001, iterations= 1000)
        model.fit(X_train, Y_train)
        Y_probability = model.predict_probability(X_test)

        # calculates TPR and FPR at multiple thresholds 
        # using scikit-learn's roc_curve function  
        fpr, tpr, _ = roc_curve(Y_test, Y_probability)

        tpr_list.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
        fpr_list.append(np.linspace(0, 1, 100))

    tpr_dataFrame = pd.DataFrame(tpr_list)
    print(tpr_dataFrame)
    fpr_dataFrame = pd.DataFrame(fpr_list)
    print(fpr_dataFrame)
    mean_tpr = np.mean(tpr_list, axis=0)
    
    mean_fpr = np.mean(fpr_list, axis=0)
    plt.plot(mean_fpr, mean_tpr, label=f"ROC Curve (AUC = {np.trapz(mean_tpr, mean_fpr):.2f})")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Average ROC Curve from 10-Fold Cross-Validation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
