import pandas as pd
import numpy as np

from sklearn import svm as SMV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def main():
    print("Welcome to SVM with Various Kernels testing.")
    print("In this model, accuracy will be collected with these 3 kernel types")
    print("linear, poly, and rbf")

    # Create dataframe for the data and remove the 1st column for y values
    file = r'./MNIST_CV.csv'
    df = pd.read_csv(file)
    print(df.info())
    X = df.iloc[:, df.columns != 'label'].values
    y = df['label'].values
    print ('Pixels: ',X,'\n\n','Labels: ',y, '\n\n')

    # Create the different model types 
    model_linear = SMV.SVC(kernel='linear')
    model_poly = SMV.SVC(kernel='poly')
    model_rbf = SMV.SVC(kernel='rbf')
    accuracy_linear = []
    accuracy_poly = []
    accuracy_rbf = []

    # Create the fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (training_index, testing_index) in enumerate(kfold.split(X)):
        X_train, X_test = X[training_index], X[testing_index]
        y_train, y_test = y[training_index], y[testing_index]
        model_linear.fit(X_train,y_train)
        model_poly.fit(X_train, y_train)
        model_rbf.fit(X_train, y_train)

        # Predict the values and append them to accuracy objects
        
        # Linear Model
        y_proba_linear = accuracy_score(y_test, model_linear.predict(X_test))
        accuracy_linear.append(y_proba_linear)
        
        # Poly Model
        y_proba_poly = accuracy_score(y_test, model_poly.predict(X_test))
        accuracy_poly.append(y_proba_poly)
        
        # RBF Model
        y_proba_rbf = accuracy_score(y_test, model_rbf.predict(X_test))
        accuracy_rbf.append(y_proba_rbf)

    # Printing Accuracies of each model 
    print('Accuracy of each Linear fold: ', accuracy_linear, '\n\n') 
    print('Accuracy of each Poly fold: ', accuracy_poly, '\n\n') 
    print('Accuracy of each RBF fold: ', accuracy_rbf, '\n\n') 

    # Calculate the average accuracy of each model
    average_accuracy_linear = np.average(accuracy_linear)
    average_accuracy_poly = np.average(accuracy_poly)
    average_accuracy_rbf = np.average(accuracy_rbf)

    # Printing Average Accuracies of each model
    print('Average Accuracy of Linear Model: ', average_accuracy_linear, "\n\n")
    print('Average Accuracy of Poly Model: ', average_accuracy_poly, '\n\n')
    print('Average Accuracy of RBF Model: ', average_accuracy_rbf, '\n\n')

    # Closing statement
    print('End of model comparison. Now exiting ...')

if __name__ == '__main__':
    main()