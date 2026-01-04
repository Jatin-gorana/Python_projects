import win32com.client as wincom

if __name__ == '__main__':
    print("--------------Welcome to RoboSpeaker-------------------")
    speak = wincom.Dispatch("SAPI.Spvoice")
    while True:
        x = input("Enter what you want me to speak: ")
        if x == 'q':
            speak.Speak("Bye bye friend!!")
            break
        command = f"{x}"
        speak.Speak(command)
