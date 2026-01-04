import os

import speech_recognition as sr
import win32com.client
import webbrowser
import datetime
from youtubesearchpython import VideosSearch
import openai_test

speaker = win32com.client.Dispatch("SAPI.SpVoice")


def say(text):
    speaker.Speak(text)


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # r.pause_threshold = 1
        print("Listening...")
        audio = r.listen(source)

        try:
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query
        except Exception as e:
            return "Some error occurred. Sorry from Jarvis"

if __name__ == '__main__':
    print('PyCharm')
    say("Hello, I'm Jarvis AI")
    query = takeCommand()

    sites = [["youtube", "https://youtube.com"], ["google", "https://google.com"], ["facebook", "https://facebook.com"],
             ["jatin's portfolio", "https://jatingorana.netlify.app/"]]

    for site in sites:
        if f"Open {site[0]}".lower() in query.lower():
            say(f"Opening {site[0]} sir...")
            webbrowser.open(site[1])

    if f"Open music".lower() in query.lower():
        musicPath = "C:/Users/jatin/Downloads/bgm.mp3"
        say(f"Opening music")
        os.system(f"start {musicPath}")

    if "the time" in query:
        strfTime = datetime.datetime.now().strftime("%H:%M:%S")
        say(f"Sir the time is {strfTime}")

    apps = [["vscode", "C:/Program Files/Microsoft VS Code/Code.exe"],
            ["pycharm", "C:/Program Files/JetBrains/PyCharm Community Edition 2022.2.3/bin/pycharm64.exe"],
            ["mongodb", "C:/Users/jatin/AppData/Local/MongoDBCompass/MongoDBCompass.exe"]]

    # query = takeCommand()
    for app in apps:
        if f"Open {app[0]}".lower() in query.lower():
            say(f"Opening {app[0]} sir...")
            os.startfile(app[1])


    def play_music_on_youtube(song):
        search = VideosSearch(song, limit=1)
        video_result = search.result()["result"][0]
        video_url = video_result["link"]
        webbrowser.open(video_url)


    if "play music" in query.lower():
        say("Which song?")
        song = takeCommand()
        play_music_on_youtube(song)


    def google_search(query):
        search_url = f"https://www.google.com/search?q={query}"
        webbrowser.open(search_url)


    if "search" in query.lower():
        say("What do you want to search for?")
        search_query = takeCommand()
        google_search(search_query)
