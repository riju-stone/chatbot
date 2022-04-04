import datetime
import webbrowser
import wikipediaapi
import pyjokes
from googlesearch import search
from core.tts import text_to_speech


def time_uitility():
    return "Right now it's {} and today it's {}".format(datetime.datetime.now().strftime("%H:%M:%S"), datetime.date.today())


def play_song_video(query):
    url = "https://www.youtube.com/results?search_query={}".format(query)
    try:
        res = "Let me see"
        webbrowser.open_new_tab(url)
    except:
        res = "Sorry could not find anything on Youtube"
    return res


def search_google(query):
    url = "https://www.google.com/search?q={}".format(query)
    try:
        webbrowser.open_new_tab(url)
    except:
        text_to_speech("Bot --> Sorry could not find anything on Google")


def net_search_utility(search_text):
    query = ' '.join(search_text.split()[1:])
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(query)
    res = page.summary[0: 200]

    if len(res) != 0:
        return res
    else:
        search_google(query)
        return "Sorry I do not know much about {}, but let me search and see what I can find...".format(
            query)


def joke_utility():
    return pyjokes.get_joke()


def evaluate_exp(search_text):
    exp = ' '.join(search_text.split()[2:])
    try:
        return "The answer is {}".format(eval(exp))
    except:
        return "Sorry I couldn't evaluate the expression {}".format(exp)
