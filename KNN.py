import pickle

import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import SpotifyConnection
import dataset


def main():
    allSong, targetList = dataset.main()
    allSong = np.array(allSong)
    targetList = np.array(targetList)

    allSong = listNormalizer(allSong)
    model(allSong, targetList)


def listNormalizer(mylist: np.ndarray):
    x_normed = mylist / mylist.max(axis=0)
    print("NORMALIZE EDILDI.")
    return x_normed


def model(allSong, targetList):
    predicted=[]
    X_train, X_test, y_train, y_test = train_test_split(allSong, targetList, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=13)
    knn.fit(X_train, y_train)
    joblib.dump(knn,'KNN.pkl')


    for i in range(len(X_test)):
        predict_me = np.array(X_test[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = knn.predict(predict_me)
        predicted.append(prediction)

    print(confusion_matrix(y_test, predicted))
    print("Accuracy of K-NN classifier on training set: {:.2f}".format(knn.score(X_train, y_train)))
    print("Accuracy of K-NN classifier on test set: {:.2f}".format(knn.score(X_test, y_test)))

    predictionSong()

def predictionSong():

    songUri = "spotify:track:03cu7r9g4b9yjTq6GHWcMG"
    if songUri.find("spotify") != -1:
        songUri = songUri[14:]

    artistName, songName, songInfo = SpotifyConnection.getSongInfo(songUri)
    allSong, targetList = dataset.main()
    allSong.append(songInfo)
    allSong = np.array(allSong)
    allSong = allSong / allSong.max(axis=0)

    mysong = allSong[-1:]
    model = joblib.load('KNN.pkl', mmap_mode='r')
    y_pred = model.predict(mysong)

    print(y_pred)
    print("Sanatçı:"+artistName)
    print("Şarkı Adı:" +songName)

    if(y_pred==[0]):
        print("THIS SONG IS NOT HIT")
    else:
        print("THIS SONG IS HIT")


def drafting(X_train,X_test,y_train,y_test):
    neighbors = np.arange(6, 18)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over K values
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Compute traning and test data accuracy
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

    # Generate plot
    plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    main()


