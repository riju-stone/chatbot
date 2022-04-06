import os
import sys
import transformers
import numpy as np
import random

from core.stt import speech_to_text
from core.tts import text_to_speech
from utils.utils import *
from local.queries import *

import streamlit as st

if __name__ == "__main__":

    st.title("""**LOCA Chatbot**""")
    st.subheader(
        """A conversational chatbot built using Transformer Model""")
    try:
        nlp = transformers.pipeline(
            "conversational", model="microsoft/DialoGPT-large")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        converse = True

        text = st.text_input("You: ", key="user_input")

        while converse:
            # if(str(sys.argv[1]) == "--speech"):
            #     text = speech_to_text()
            # elif(str(sys.argv[1]) == "--text"):

            if any(i in text for i in conv_end):
                res = np.random.choice(conv_end_res)
                converse = False

            elif any(i in text for i in time_query):
                res = time_uitility()

            elif any(i in text for i in search_query):
                res = net_search_utility(text)

            elif any(i in text for i in joke_query):
                res = joke_utility()

            elif any(i in text for i in eval_query):
                res = evaluate_exp(text)

            elif any(i in text for i in identity_query):
                res = np.random.choice(identity_res)

            elif any(i in text for i in youtube_query):
                res = play_song_video(text)

            else:

                if text == "ERROR":
                    res = np.random.choice(failed_res)
                else:
                    with st.spinner("Thinking..."):
                        chat = nlp(transformers.Conversation(
                            text), pad_token_id=50256)
                        res = str(chat)
                        res = res[res.find("bot >>")+6:].strip()

                text_to_speech(res)
    except:
        pass
