from gtts import gTTS
import pyglet
import os
from time import sleep
import streamlit as st

filename = 'speech.mp3'


def text_to_speech(text):
    st.text_area("Bot :", value=text, height=200, max_chars=None, key=None)

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(filename)

    speech = pyglet.media.load(filename, streaming=False)
    speech.play()

    sleep(speech.duration)
    os.remove(filename)
