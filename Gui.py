from tkinter import *
import joblib
from pandas import np
from tensorflow.python.keras.models import model_from_json
import SpotifyConnection
import dataset


def predictionSong(songUri: str, songNameLabel: Label, artistNameLabel: Label, hitRateLabel: Label, resultLabel: Label,
                   decisionHitLabel: Label, decisionResult: Label, knnHitLabel: Label, knnResult: Label, svmHitLabel:Label, svmResult: Label,
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

    songNameLabel['text'] = "ŞARKI İSMİ: " + songName
    artistNameLabel['text'] = "SANATÇI İSMİ: " + artistName

    predict, predictions, result = neuralPrediction(mysong)
    hitRateLabel['text'] = "POPÜLER OLMA İHTİMALİ: % " + "%.2f" % (predict[0][0] * 100)+" "+str(predictions[0])
    resultLabel['text'] = "SONUÇ: " + result
    print("Hit Rate: %" + "%.2f" % (predict[0][0] * 100))

    predict, result = decisionTreePrediction(mysong)
    decisionHitLabel['text'] = "POPÜLERLİK : " + str(predict)
    decisionResult['text'] = "SONUÇ: " + result

    predict, result = knnPrediction(mysong)
    knnHitLabel['text'] = "POPÜLERLİK : " + str(predict)
    knnResult['text'] = "SONUÇ: " + result

    predict, result = svmPrediction(mysong)
    svmHitLabel['text'] = "POPÜLERLİK : " + str(predict)
    svmResult['text'] = "SONUÇ: " + result

    predict, result = bayesPrediction(mysong)
    bayesHitLabel['text'] = "POPÜLERLİK : " + str(predict)
    bayesResult['text'] = "SONUÇ: " + result


def neuralPrediction(mysong):
    model = readModelFromJSON()

    predict = model.predict(mysong)
    predictions= model.predict_classes(mysong)
    print(predictions[0])
    print(predict)
    if predict >= 0.50:
        print("THIS SONG IS HIT!")
        result = "BU ŞARKI POPÜLER!"
        return predict, predictions, result
    else:
        print("THIS SONG IS NOT HIT!")
        result = "BU ŞARKI POPÜLER DEĞİL!"
        return predict, predictions, result


def decisionTreePrediction(mysong):
    model = joblib.load('decisionTree.pkl', mmap_mode='r')
    predict = model.predict(mysong)
    if (predict == [0]):
        print("THIS SONG IS NOT HIT")
        result = "BU ŞARKI POPÜLER DEĞİL!"
        return predict, result
    else:
        print("THIS SONG IS HIT")
        result = "BU ŞARKI POPÜLER!"
        return predict, result


def knnPrediction(mysong):
    model = joblib.load('KNN.pkl', mmap_mode='r')
    predict = model.predict(mysong)
    if (predict == [0]):
        print("THIS SONG IS NOT HIT")
        result = "BU ŞARKI POPÜLER DEĞİL!"
        return predict, result
    else:
        print("THIS SONG IS HIT")
        result = "BU ŞARKI POPÜLER!"
        return predict, result


def svmPrediction(mysong):
    model = joblib.load('SVM.pkl', mmap_mode='r')
    predict = model.predict(mysong)
    if (predict == [0]):
        print("THIS SONG IS NOT HIT")
        result = "BU ŞARKI POPÜLER DEĞİL!"
        return predict, result
    else:
        print("THIS SONG IS HIT")
        result = "BU ŞARKI POPÜLER!"
        return predict, result


def bayesPrediction(mysong):
    model = joblib.load('BNB.pkl', mmap_mode='r')
    predict = model.predict(mysong)
    if (predict == [0]):
        print("THIS SONG IS NOT HIT")
        result = "BU ŞARKI POPÜLER DEĞİL!"
        return predict, result
    else:
        print("THIS SONG IS HIT")
        result = "BU ŞARKI POPÜLER!"
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
    root.geometry("400x500")
    root.title("Popüler Şarkı Tahmini")
    root.resizable(False,False)
    frame = Frame(root, background='white')
    frame.pack(fill=BOTH, expand=True)
    header = Label(frame, text="---------------- POPÜLER ŞARKI TAHMİN EDİCİ ----------------", font='Helvetica 12 bold',bg='white')
    header.pack()
    enterText = Label(frame, text="Lütfen şarkınızın Spotify URI bilgisini giriniz", font='Helvetica 8 bold',bg='white')
    enterText.pack()
    getUri = Entry(frame, width=40)
    getUri.pack()
    songName = Label(frame, text="ŞARKI İSMİ: - ",bg='white')
    artistName = Label(frame, text="SANATÇI İSMİ: - ",bg='white')
    hitRate = Label(frame, text="POPÜLER OLMA İHTİMALİ: - ",bg='white')
    result = Label(frame, text="SONUÇ: - ",bg='white')
    decisionHitRate = Label(frame, text="POPÜLERLİK: - ",bg='white')
    decisionResult = Label(frame, text="SONUÇ: - ",bg='white')
    knnHitRate = Label(frame, text="POPÜLERLİK: - ",bg='white')
    knnResult = Label(frame, text="SONUÇ: - ",bg='white')
    svmHitRate = Label(frame, text="POPÜLERLİK: - ",bg='white')
    svmResult = Label(frame, text="SONUÇ: - ",bg='white')
    bayesHitRate = Label(frame, text="POPÜLERLİK: - ",bg='white')
    bayesResult = Label(frame, text="SONUÇ: - ",bg='white')
    searchButton = Button(frame, text="Tahmin Et!", command=(
        lambda: predictionSong(getUri.get(), songName, artistName, hitRate, result,decisionHitRate,
                               decisionResult,knnHitRate, knnResult, svmHitRate, svmResult,bayesHitRate,
                               bayesResult)),bg='#33FFFF', width=20)
    searchButton.pack()
    songName.pack()
    artistName.pack()
    neuralText = Label(frame, text="Yapay Sinir Ağı", font='Helvetica 10 bold',bg='white')
    neuralText.pack()
    hitRate.pack()
    result.pack()
    decisionText = Label(frame, text="Karar Ağacı", font='Helvetica 10 bold',bg='white')
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
