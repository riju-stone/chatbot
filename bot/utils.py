import datetime
from numpy.core.numeric import identity
import wikipedia
import yake
import pyjokes

identity_query = ["who are you", "your name"]
identity_res = ["My name is LOCA or the Least Obfuscated Chatbot Aplication",
                "I'm the Least Obfuscated Chatbot Aplication or you can call me LOCA"]
conv_end = ["goodbye", "bye", "see you", "tata", "see you"]
conv_end_res = ["Tata", "Have a good day", "Bye",
                "Goodbye", "Hope to meet soon", "Peace out!"]
failed_res = ["I beg your pardon.", "Please repeat.", "Sorry I couldn't get you.",
              "Can you speak up ?", "I didn't hear you. Sorry say it again."]
search_query = ["define", "explain"]
time_query = ["the time", "the date"]
joke_query = ["tell me a joke", "tell a joke", "do you know any jokes"]


def time_uitility():
    return "Right now it's {} and today it's {}".format(datetime.datetime.now().strftime("%H:%M:%S"), datetime.date.today())


def net_search_utility(search_text):

    extract_model = yake.KeywordExtractor(search_text)
    keywords = extract_model.extract_keywords(search_text)

    query = keywords[0][0]

    try:
        res = wikipedia.summary(query, sentences=2)
    except:
        res = "Sorry Couldn't find anything with '{}'".format(query)

    return res


def joke_utility():
    return pyjokes.get_joke()
