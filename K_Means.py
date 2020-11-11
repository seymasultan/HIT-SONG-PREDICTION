import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
    X_train, X_test, y_train, y_test = train_test_split(allSong, targetList, test_size=0.2)
    kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10, max_iter=300)
    kmeans.fit(X_train, y_train)
    saved_model = pickle.dumps(kmeans)

    sifirSifir = 0
    sifirBir = 0
    birSifir = 0
    birBir = 0

    for i in range(len(X_test)):
        predict_me = np.array(X_test[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0]:
            if (y_test[i] == 0):
                sifirSifir = sifirSifir + 1
            else:
                sifirBir = sifirBir + 1
        else:
            if (y_test[i] == 0):
                birSifir = birSifir + 1
            else:
                birBir = birBir + 1

    print("sifirSifir:" + str(sifirSifir))
    print("sifirBir:" + str(sifirBir))
    print("birSifir:" + str(birSifir))
    print("birBir:" + str(birBir))
    print(kmeans.cluster_centers_)

    predictionSong(saved_model)


def predictionSong(saved_model):
    songUri = "spotify:track:03cu7r9g4b9yjTq6GHWcMG"
    if songUri.find("spotify") != -1:
        songUri = songUri[14:]

    artistName, songName, songInfo = SpotifyConnection.getSongInfo(songUri)
    allSong, targetList = dataset.main()
    allSong.append(songInfo)
    allSong = np.array(allSong)
    allSong = allSong / allSong.max(axis=0)

    mySong = allSong[-1:]
    k_means_from_pickle = pickle.loads(saved_model)
    y_pred = k_means_from_pickle.predict(mySong)
    print(y_pred)


def drafting(X_train, y_train):
    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(i)
        kmeans.fit(X_train, y_train)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 10), wcss)
    plt.xlabel('K DEĞERİ')
    plt.ylabel('WCSS')
    plt.show()


if __name__ == '__main__':
    main()
