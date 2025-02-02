import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

import SpotifyConnection
import dataset
import numpy as np


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
    dtr = DecisionTreeClassifier(random_state=0, criterion = "gini")
    # Makinemizi eğittik
    dtr.fit(X_train, y_train)
    joblib.dump(dtr,'decisionTree.pkl')

    for i in range(len(X_test)):
        predict_me = np.array(X_test[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = dtr.predict(predict_me)
        predicted.append(prediction)

    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))
    print("Accuracy of Decision Tree classifier on training set: {:.2f}".format(dtr.score(X_train, y_train)))
    print("Accuracy of Decision Tree classifier on test set: {:.2f}".format(dtr.score(X_test, y_test)))

    predictionSong()


def predictionSong():

    songUri = "spotify:track:1aaI0imelqLqye35922oMD"
    if songUri.find("spotify") != -1:
        songUri = songUri[14:]

    artistName, songName, songInfo = SpotifyConnection.getSongInfo(songUri)
    allSong, targetList = dataset.main()
    allSong.append(songInfo)
    allSong = np.array(allSong)
    allSong = allSong / allSong.max(axis=0)

    mySong = allSong[-1:]
    model = joblib.load('decisionTree.pkl',mmap_mode='r')
    y_pred = model.predict(mySong)

    print(y_pred)
    print("Sanatçı:" + artistName)
    print("Şarkı Adı:" + songName)

    if (y_pred == [0]):
        print("THIS SONG IS NOT HIT")
    else:
        print("THIS SONG IS HIT")


if __name__ == '__main__':
    main()