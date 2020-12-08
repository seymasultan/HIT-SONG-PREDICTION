from tkinter import *
import joblib
from pandas import np
from tensorflow.python.keras.models import model_from_json
import SpotifyConnection
import dataset


def predictionSong(songUri: str, songNameLabel: Label, artistNameLabel: Label, hitRateLabel: Label, resultLabel: Label,
                   decisionHitLabel: Label,
                   decisionResult: Label, knnHitLabel: Label, knnResult: Label, svmHitLabel: Label, svmResult: Label,
                   bayesHitLabel: Label, bayesResult: Label):
    if songUri.find("spotify") != -1:
        songUri = songUri[14:]

    # yeni şarkının veri setine eklenmesi ve normalize edilmesi
    artistName, songName, songInfo = SpotifyConnection.getSongInfo(songUri)
    allSong, targetList = dataset.main()
    allSong.append(songInfo)
    allSong = np.array(allSong)
    allSong = allSong / allSong.max(axis=0)

    # veri setinden şarkının alınması ve tahmini
    mysong = allSong[-1:]
    print("*************HIT SONG PREDICTION*************")
    print("Song Name: " + songName)
    print("Artist Name:" + artistName)

    songNameLabel['text'] = "SONG NAME: " + songName
    artistNameLabel['text'] = "ARTIST NAME: " + artistName

    predict, result = neuralPredict(mysong)
    hitRateLabel['text'] = "HIT RATE: % " + "%.2f" % (predict[0][0] * 100)
    resultLabel['text'] = "RESULT: " + result
    print("Hit Rate: %" + "%.2f" % (predict[0][0] * 100))

    predict, result = decisionTreePredict(mysong)
    decisionHitLabel['text'] = "HIT : " + str(predict)
    decisionResult['text'] = "RESULT: " + result

    predict, result = knnPrediction(mysong)
    knnHitLabel['text'] = "HIT : " + str(predict)
    knnResult['text'] = "RESULT: " + result

    predict, result = svmPrediction(mysong)
    svmHitLabel['text'] = "HIT : " + str(predict)
    svmResult['text'] = "RESULT: " + result

    predict, result = bayesPrediction(mysong)
    bayesHitLabel['text'] = "HIT : " + str(predict)
    bayesResult['text'] = "RESULT: " + result


def neuralPredict(mysong):
    model = readModelFromJSON()

    predict = model.predict(mysong)

    if predict >= 0.7:
        print("THIS SONG IS HIT!")
        result = "THIS SONG IS HIT!"
        return predict, result
    else:
        print("THIS SONG IS NOT HIT!")
        result = "THIS SONG IS NOT HIT!"
        return predict, result


def decisionTreePredict(mysong):
    model = joblib.load('decisionTree.pkl', mmap_mode='r')
    predict = model.predict(mysong)
    if (predict == [0]):
        print("THIS SONG IS NOT HIT")
        result = "THIS SONG IS NOT HIT!"
        return predict, result
    else:
        print("THIS SONG IS HIT")
        result = "THIS SONG IS HIT!"
        return predict, result


def knnPrediction(mysong):
    model = joblib.load('KNN.pkl', mmap_mode='r')
    predict = model.predict(mysong)
    if (predict == [0]):
        print("THIS SONG IS NOT HIT")
        result = "THIS SONG IS NOT HIT!"
        return predict, result
    else:
        print("THIS SONG IS HIT")
        result = "THIS SONG IS HIT!"
        return predict, result


def svmPrediction(mysong):
    model = joblib.load('KNN.pkl', mmap_mode='r')
    predict = model.predict(mysong)
    if (predict == [0]):
        print("THIS SONG IS NOT HIT")
        result = "THIS SONG IS NOT HIT!"
        return predict, result
    else:
        print("THIS SONG IS HIT")
        result = "THIS SONG IS HIT!"
        return predict, result


def bayesPrediction(mysong):
    model = joblib.load('BNB.pkl', mmap_mode='r')
    predict = model.predict(mysong)
    if (predict == [0]):
        print("THIS SONG IS NOT HIT")
        result = "THIS SONG IS NOT HIT!"
        return predict, result
    else:
        print("THIS SONG IS HIT")
        result = "THIS SONG IS HIT!"
        return predict, result


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


def gui():
    root = Tk()
    root.geometry("700x500")
    root.title("Hit Song Prediction")
    root.resizable(False,False)
    frame = Frame(root, background='white')
    frame.pack(fill=BOTH, expand=True)
    header = Label(frame, text="---------------- HIT PREDICTOR ----------------", font='Helvetica 12 bold',bg='white')
    header.pack()
    enterText = Label(frame, text="Please enter Spotify URI of your song", font='Helvetica 8 bold',bg='white')
    enterText.pack()
    getUri = Entry(frame, width=40)
    getUri.pack()
    songName = Label(frame, text="SONG NAME: - ",bg='white')
    artistName = Label(frame, text="ARTIST NAME: - ",bg='white')
    hitRate = Label(frame, text="HIT RATE: - ",bg='white')
    result = Label(frame, text="RESULT: - ",bg='white')
    decisionHitRate = Label(frame, text="HIT: - ",bg='white')
    decisionResult = Label(frame, text="RESULT: - ",bg='white')
    knnHitRate = Label(frame, text="HIT: - ",bg='white')
    knnResult = Label(frame, text="RESULT: - ",bg='white')
    svmHitRate = Label(frame, text="HIT: - ",bg='white')
    svmResult = Label(frame, text="RESULT: - ",bg='white')
    bayesHitRate = Label(frame, text="HIT: - ",bg='white')
    bayesResult = Label(frame, text="RESULT: - ",bg='white')
    searchButton = Button(frame, text="Predict!", command=(
        lambda: predictionSong(getUri.get(), songName, artistName, hitRate, result, decisionHitRate,
                               decisionResult, knnHitRate, knnResult, svmHitRate, svmResult, bayesHitRate,
                               bayesResult)),bg='#33FFFF', width=20)
    searchButton.pack()
    songName.pack()
    artistName.pack()
    neuralText = Label(frame, text="Neural Network", font='Helvetica 10 bold',bg='white')
    neuralText.pack()
    hitRate.pack()
    result.pack()
    decisionText = Label(frame, text="Decision Tree", font='Helvetica 10 bold',bg='white')
    decisionText.pack()
    decisionHitRate.pack()
    decisionResult.pack()
    knnText = Label(frame, text="KNN", font='Helvetica 10 bold',bg='white')
    knnText.pack()
    knnHitRate.pack()
    knnResult.pack()
    svmText = Label(frame, text="SVM", font='Helvetica 10 bold',bg='white')
    svmText.pack()
    svmHitRate.pack()
    svmResult.pack()
    bayesText = Label(frame, text="Naive Bayes", font='Helvetica 10 bold',bg='white')
    bayesText.pack()
    bayesHitRate.pack()
    bayesResult.pack()
    root.mainloop()


if __name__ == '__main__':
    gui()
