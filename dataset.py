import pandas as pd
import math


def main():
    allsong = []
    targetList = []

    dataset = pd.read_excel('allDataset.xlsx', sheet_name='Worksheet')
    dataset = dataset.drop(columns=['track', 'artist', 'uri', 'chorus_hit', 'sections'])

    # verisetinden yanlış olan satırlar temizlenir
    for row in dataset.values:
        if type(row[0]) is float and type(row[1]) is float and not math.isnan(row[0]):
            song = []
            for element in row[0:13]:
                song.append(element)
            targetList.append(row[13])
            allsong.append(song)

    print("Dosya Okuma Tamamlandı")
    print("Toplam Şarkı Sayısı:" + str(len(allsong)))
    print("Toplam Sütun Sayısı:" + str(len(allsong[0])))

    return allsong, targetList

if __name__ == '__main__':
    main()
