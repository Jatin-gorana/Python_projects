import requests
import json
import win32com.client as wincom
speak = wincom.Dispatch("Sapi.SpVoice")

city = input("Enter the name of the city: \n")

# url = "http://api.weatherapi.com/v1/current.json?key=b13989793f184149a91141538230103&q=London&aqi=yes"

url1 = f"http://api.weatherapi.com/v1/current.json?key=b13989793f184149a91141538230103&q={city}&aqi=yes"

r = requests.get(url1)
# print(r.text)
wdic = json.loads(r.text)
work = input("What weather info do you want:\n"
             "1. Press 1 for longitude and latitude\n"
             "2. Press 2 for region and country\n"
             "3. Press 3 for temperature in celcius and farenheit\n"
             "4. Press 4 for wind speed\n"
             "5. Press 5 for cloud and humidity\n")
if work == '1':
    longitude = wdic["location"]["lon"]
    latitude = wdic["location"]["lat"]
    print(f"The longitude and latitude in {city} is {longitude} and {latitude} respectively")
    speak.Speak(f"The longitude and latitude in {city} is {longitude} and {latitude} respectively")
elif work == '2':
    reg = wdic["location"]["region"]
    country = wdic["location"]["country"]
    print(f"The {city} is located in {reg} region. Country is {country}")
    speak.Speak(f"The {city} is located in {reg} region. Country is {country}")
elif work == '3':
    c = wdic["current"]["temp_c"]
    f = wdic["current"]["temp_f"]
    print(f"The temperature in {city} is {c} degree celcius or {f} degree farenheit")
    speak.Speak(f"The temperature in {city} is {c} degree celcius or {f} degree farenheit")
elif work == '4':
    wsmph = wdic["current"]["wind_mph"]
    wskph = wdic["current"]["wind_kph"]
    print(f"{city} has windspeed of {wskph} kilometers per hour or {wsmph} miles per hour")
    speak.Speak(f"{city} has windspeed of {wskph} kilometers per hour or {wsmph} miles per hour")
elif work == '5':
    cld = wdic["current"]["cloud"]
    hmdt = wdic["current"]["humidity"]
    print(f"{city} has {cld} cloud and {hmdt} humidity")
    speak.Speak(f"{city} has {cld} cloud and {hmdt} humidity")
else:
    print("Invalid input!!!!.Try again.")


# w = wdic["current"]["temp_c"]
# reg = wdic["location"]["region"]
# speak.Speak(f"The current weather in {city} region {reg} is {w} degrees")

