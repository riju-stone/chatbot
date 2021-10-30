import os
import transformers
import numpy as np

from stt import speech_to_text
from tts import text_to_speech
from utils import conv_end, conv_end_res, failed_res, net_search_utility, time_uitility

if __name__ == "__main__":
    nlp = transformers.pipeline(
        "conversational", model="facebook/blenderbot_small-90M")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    processing = True

    while processing:
        text = speech_to_text()

        if any(i in text for i in conv_end):
            res = np.random.choice(conv_end_res)
            processing = False

        elif any(i in text for i in ["time", "date"]):
            res = time_uitility()

        elif any(i in text for i in ["define", "explain", "what can you tell me about"]):
            res = net_search_utility(text)

        else:
            if text == "ERROR":
                res = np.random.choice(failed_res)
            else:
                chat = nlp(transformers.Conversation(text), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >>")+6:].strip()

        text_to_speech(res)
