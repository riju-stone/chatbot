from gtts import gTTS
import pyglet
import os
from time import sleep

filename = 'speech.mp3'

def text_to_speech(text):
    print("Bot --> ", text)

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(filename)

    speech = pyglet.media.load(filename, streaming=False)
    speech.play()

    sleep(speech.duration)
    os.remove(filename)
