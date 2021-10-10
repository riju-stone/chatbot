import pickle
import collections


def process():
    with open("processed_scripts/intent_query.pkl", 'rb') as fp:
        intent_query_dict = pickle.load(fp)

    with open("processed_scripts/intent_response.pkl", 'rb') as fp:
        intent_response_dict = pickle.load(fp)

    qa_ordered_dict = collections.OrderedDict()

    for intent in intent_response_dict:
        queries = intent_query_dict[intent]
        for query in queries:
            qa_ordered_dict[query] = intent_response_dict[intent]

    with open("processed_scripts/bot_profile.pkl", "wb") as fp:
        pickle.dump(qa_ordered_dict, fp)
