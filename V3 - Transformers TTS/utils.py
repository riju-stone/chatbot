import datetime
import wikipedia

conv_end = ["goodbye", "bye", "see you", "tata", "exit", "close"]
conv_end_res = ["Tata", "Have a good day", "Bye",
                "Goodbye", "Hope to meet soon", "Peace out!"]
failed_res = ["I bet your pardon.", "Please repeat.", "Sorry I couldn't get you.",
              "Can you speak up ?", "I didn't hear you. Sorry say it again."]
search_query = ["define", "explain",
                "what can you tell me about", "what do you mean by"]
time_query = ["the time", "the date"]


def time_uitility():
    return "Right now it's {} and today it's {}".format(datetime.datetime.now().strftime("%H:%M:%S"), datetime.date.today())


def net_search_utility(search_text):
    try:
        res = wikipedia.summary(search_text, sentences=4)
    except:
        res = "Sorry Couldn't find anything with '{}'".format(search_text)

    return res
