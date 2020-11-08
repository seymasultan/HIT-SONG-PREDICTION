import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)


# Şarkının uri'si girildiğinde şarkıya ait features_list listesindeki özelliklerinin okunması


def getSongInfo(songUri):
    features_list = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
                     "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]

    track = sp.audio_features(songUri)
    trackInfo = sp.track(songUri)
    artists = trackInfo["artists"]
    artistName = artists[0]["name"]
    songName = trackInfo["name"]

    songInfo = []

    for feature in features_list:
        songInfo.append(track[0][feature])

    print("Song read completed")
    print(songInfo)
    print(artistName)
    print(songName)

    return artistName, songName, songInfo


if __name__ == '__main__':
    getSongInfo("4WQWrSXYLnwwcmdNk8dYqN")
