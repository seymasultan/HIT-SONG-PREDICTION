from tkinter import Label

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import model_from_json
import numpy as np
import dataset
import SpotifyConnection


# Urisi girilen şarkının hit tahmininin gerçekleştirilmesi
def predictionSong(songUri: str,songNameLabel: Label, artistNameLabel: Label, hitRateLabel: Label, resultLabel: Label):

    # Girilin uri "spotify:track:URI" şeklinde ise URI kısmını ayırmak için kullanılır.
    if songUri.find("spotify") != -1:
        songUri = songUri[14:]

    # yeni şarkının veri setine eklenmesi ve normalize edilmesi
    artistName, songName, songInfo = SpotifyConnection.getSongInfo(songUri)
    allSong, targetList = dataset.main()
    allSong.append(songInfo)
    allSong = np.array(allSong)
    allSong = allSong / allSong.max(axis=0)

    # Modelin hazırlanması
    model = readModelFromJSON()

    #veri setinden şarkının alınması ve tahmini
    mysong = allSong[-1:]
    predict = model.predict(mysong)

    songNameLabel['text'] = "SONG NAME: " + songName
    artistNameLabel['text'] = "ARTIST NAME: " + artistName
    hitRateLabel['text'] = "HIT RATE: % " + "%.2f" % (predict[0][0] * 100)

    print("*************HIT SONG PREDICTION*************")
    print("Song Name: " +songName)
    print("Artist Name:" +artistName)
    print("Hit Rate: %" + "%.2f" % (predict[0][0] * 100))

    if predict >= 0.7:
        print("SONG IS HIT!")
        resultLabel['text'] = "THIS SONG IS HIT!"
    else:
        print("SONG IS NOT HIT!")
        resultLabel['text'] = "THIS SONG IS NOT HIT!"

# Modelin dosya sisteminden okunmasını sağlar.
def readModelFromJSON():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Model is readed.")
    return loaded_model


if __name__ == '__main__':
 predictionSong("spotify:track:6ICZQc8fr18dK37RN6R2YL")

