import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

class LogitRegression():
    def __init__ (self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
    def fit(self, X, y):
        self.training, self.features = X.shape
        self.weight = np.random.randn(self.features) * 0.01
        self.bias = 0
        self.X = X
        self.y = y 

        for i in range(self.iterations) :
            self.update_weights()
        return self
    def update_weights (self):
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
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.weight ) + self.bias ) ) )         
        Y = np.where( Z > 0.5, 1, 0 )         
        return Y 

    
def Sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def LogisticRegression(X, y, learning_rate=0.01, num_iterations=500, epsilon=1e-10):
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)  # 1D array for coefficients
    costs = []

    for i in range(num_iterations):
        # Compute the sigmoid function on the linear model output
        logistic = Sigmoid(X.dot(coefficients))
        logistic = np.clip(logistic, epsilon, 1 - epsilon)

        # Calculate the gradient and update coefficients
        gradient = X.T.dot(y - logistic) / n_samples
        coefficients += learning_rate * gradient

        # Calculate cost for logging
        cost = -np.mean(y * np.log(logistic) + (1 - y) * np.log(1 - logistic))
        costs.append(cost)

    return coefficients, costs

def predictions(X, coefficients):
    logistic = Sigmoid(X.dot(coefficients))
    return [1 if i > 0.5 else 0 for i in logistic]

def KFoldCrossValidation(feature_columns, y_values, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for i, (training, testing) in enumerate(kf.split(feature_columns)):
        # Align indices by resetting the index
        X_training = feature_columns.iloc[training].reset_index(drop=True)
        X_test = feature_columns.iloc[testing].reset_index(drop=True)
        y_training = y_values.iloc[training].reset_index(drop=True)
        y_test = y_values.iloc[testing].reset_index(drop=True)

        # Train the model only on the training fold
        fold_coefficients, fold_costs = LogisticRegression(X_training, y_training)

        # Use the trained model to predict on the test fold
        fold_predictions = predictions(X_test, fold_coefficients)
        
        # Check that predictions match y_test length
        if len(fold_predictions) != len(y_test):
            print(f"Fold {i + 1}: Prediction length mismatch.")
            continue

        # Calculate accuracy
        accuracy = np.mean(np.array(fold_predictions) == y_test.values) * 100
        print(f"Fold {i + 1}: Accuracy = {accuracy:.2f}%")


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
    fold_accuracies = []
    for fold, (training_index, testing_index) in enumerate(kfold.split(X_values), 1):
        X_train, X_test = X_values[training_index], X_values[testing_index]
        Y_train, Y_test = y_values[training_index], y_values[testing_index]
        model = LogitRegression(learning_rate= 0.001, iterations= 1000)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        accuracy = np.mean(Y_test == y_pred) * 100
        fold_accuracies.append(accuracy)
        print(f"Fold {fold}: Accuracy = {accuracy:.2f}%")

    average_accuracy = np.mean(fold_accuracies)

    print("Accuracy on test set by our model: ", average_accuracy)

    '''
    print("Starting K-fold cross validation ... ")
    KFoldCrossValidation(X_values, y_values, 10)
    '''
if __name__ == "__main__":
    main()
