import speech_recognition as sr


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as mic:
        print("Say Something. I'm all ears...")
        audio = recognizer.listen(mic, timeout=10)

    try:
        text = recognizer.recognize_google(audio)
        print("User --> ", text)
    except:
        text = "ERROR"

    return text
