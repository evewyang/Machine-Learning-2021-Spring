from sys import argv
# import modules
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split


def load_data():
    #Question (a)
    # load iris dataset
    iris = ds.load_iris()

    # assign iris features to X, an array of shape (150,4)
    # assign iris labels to y, an array of shape (150,)
    X = iris['data']
    y = iris['target']

    # calculate elements in each class
    label_id,label_count = np.unique(y, return_counts=True)
    # print out the result
    print("-------Question 2(a)------")
    for label, count in zip(label_id, label_count):
        print("Class %d has %d elements."%(label, count))
    return X, y, iris["feature_names"]


def run_a_to_d(X, y, feature_names):
    #Question (b)
    print("-------Question 2(b)------")
    # initialize the knn model
    model_knn = KNeighborsClassifier(n_neighbors=1)
    model_knn.fit(X, y)

    # calculate prediction accuracy
    predicted_label = model_knn.predict(X)
    # print out the accuracy
    print("The accuracy is %d%%"%(np.mean(predicted_label == y)*100))
    
    #Question (c)
    print("-------Question 2(c)------")
    # split the dataset into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=0)

    # try different value of k from 1 to 50
    K = 50
    train_accu = np.zeros(50)
    test_accu = np.zeros(50)
    for i in range(1,K+1):
        # initialize the model
        model_knn = KNeighborsClassifier(n_neighbors=i)
        # fit the data
        model_knn.fit(X_train,y_train)
        # store training accuracy in train_accu
        predicted_train_label = model_knn.predict(X_train)
        train_accu[i-1] = np.mean(predicted_train_label == y_train)
        # store validation accuracy in test_acc
        predicted_test_label = model_knn.predict(X_test)
        test_accu[i-1] = np.mean(predicted_test_label == y_test)

    # plot the training accuracy and test accuracy against k
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    x_range = np.linspace(1, K, num=K)
    plt.plot(x_range, train_accu, label='training')
    plt.plot(x_range, test_accu, label='test')
    plt.legend()
    plt.savefig("wy818_accuracy.jpg")

    # find the optimal k value
    optimal_k = test_accu.argmax() + 1#the test_accu index start from 0
    # print out the optimal k
    print("The optimal k value is: %d"%(optimal_k))

    #Question (d)
    print("-------Question 2(d)------")
    # check the order of the features
    print(feature_names)
    # match the input values with the feature names
    features = np.array([[3.8, 5.0, 1.2, 4.1]])

    model_knn = KNeighborsClassifier(n_neighbors=optimal_k)
    model_knn.fit(X,y)
    # make prediction
    predicted_label = model_knn.predict(features)
    # print out the prediction result
    print("Predicted class of this plant: %d"%(predicted_label[0]))

if __name__=="__main__":
    if argv[1] == "run":
        run_a_to_d(*load_data())
