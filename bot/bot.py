import os
import sys
import transformers
import numpy as np

from stt import speech_to_text
from tts import text_to_speech
from utils import *

if __name__ == "__main__":
    nlp = transformers.pipeline(
        "conversational", model="facebook/blenderbot_small-90M")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    converse = True

    while converse:
        if(str(sys.argv[1]) == "--speech"):
            text = speech_to_text()
        elif(str(sys.argv[1]) == "--text"):
            text = input("Write Something...\n")

        if any(i in text for i in conv_end):
            res = np.random.choice(conv_end_res)
            converse = False

        elif any(i in text for i in time_query):
            res = time_uitility()

        elif any(i in text for i in search_query):
            res = net_search_utility(text)

        elif any(i in text for i in joke_query):
            res = joke_utility()

        elif any(i in text for i in identity_query):
            res = np.random.choice(identity_res)
        else:
            if text == "ERROR":
                res = np.random.choice(failed_res)
            else:
                chat = nlp(transformers.Conversation(text), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >>")+6:].strip()

        text_to_speech(res)
