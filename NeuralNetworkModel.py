from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import CSVLogger
import dataset
import numpy as np
from matplotlib import pyplot as plt

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
    X_train, X_valid, y_train, y_valid = train_test_split(allSong, targetList, test_size=0.2)

    model = Sequential()
    model.add(Dense(32, input_dim=13, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    csv_logger = CSVLogger("model.csv", append=True)
    history=model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=64, verbose=2,
              callbacks=[csv_logger])
    scores = model.evaluate(X_train, y_train, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    drawingGraph(history)
    saveModelToJSON(model, "model")


# Modeli kaydeder.
def saveModelToJSON(model, name: str):
    model_json = model.to_json()
    with open("./" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./model.h5")
    print("Model kaydedildi.")

def drawingGraph(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('Training Loss and Accuracy')
    plt.ylabel('Accuracy/Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_acc','train_loss'], loc='lower left')
    plt.show()

if __name__ == '__main__':
    main()
