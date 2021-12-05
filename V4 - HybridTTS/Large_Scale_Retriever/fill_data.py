import sqlite3
import csv
import pickle
import numpy as np
import sys
sys.path.append("../")  # nopep8
from Sentence_Encoder.query_encoder import encode
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

conn = sqlite3.connect("database/reddit.db")

conn.execute('''CREATE TABLE queries ( id text, title text, embedding BLOB)''')
conn.execute(
    '''CREATE TABLE responses (thread_id text, parent_id text, comment text)''')


def save_queries(queries, query_index):
    global conn

    rows = []
    contexts = ["" for _ in queries]
    embeddings = encode(queries, contexts)
    embeddings = embeddings.tolist()

    for query, query_id, embedding in zip(queries, query_index, embeddings):
        embedding = np.asarray(embedding, np.float32)
        rows.append((query_id, query, embedding))

    conn.executemany('INSERT INTO queries VALUES (?, ?, ?)', rows)


def save_responses(comments, parent_index, thread_index):
    global conn
    rows = []

    for thread_id, parent_id, comment in zip(thread_index, parent_index, comments):
        rows.append((thread_id, parent_id, comment))

    conn.executemany('INSERT INTO responses VALUES (?, ?, ?)', rows)


filepaths_q = [
    'data/askredditq.csv',
    'data/adviceq.csv',
    'data/askphilosophyq.csv',
    'data/askscienceq.csv',
    'data/casualq.csv',
    'data/eli5q.csv',
    'data/mlq.csv'
]

filepaths_a = [
    'data/askreddita.csv',
    'data/advicea.csv',
    'data/askphilosophya.csv',
    'data/asksciencea.csv',
    'data/casuala.csv',
    'data/eli5a.csv',
    'data/mla.csv'
]

for filename_q, filename_a in zip(filepaths_q, filepaths_a):
    queries = []
    responses = []
    query_index = []
    response_thread_index = []
    response_parent_index = []

    print("\nProcessing {} and {}... \n".format(filename_q, filename_a))

    comment_thread_index = {}
    thread_index = {}

    with open(filename_a, newline='', encoding='cp850') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            id = str(row['id'])
            thread_id = str(row['link_id'])[3:]
            parent_id = str(row['parent_id'])[3:]
            comment = str(row['body'])

            if len(comment.split(" ")) <= 300:
                if parent_id == thread_id:
                    if thread_id not in comment_thread_index:
                        comment_thread_index[thread_id] = 1

    with open(filename_q, newline='', encoding='cp850') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            id = str(row['id'])
            title = str(row['title'])

            if len(title.split(" ")) <= 200:
                if id not in thread_index:
                    thread_index[id] = 1

    print("\nProcessing Queries\n")
    with open(filename_q, newline='', encoding='cp850') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            id = str(row['id'])
            title = str(row['title'])

            if len(title.split(" ")) <= 200:
                if id in comment_thread_index:
                    queries.append(title)
                    query_index.append(id)
                    if len(queries) > 500:
                        print(i)
                        save_queries(queries, query_index)
                        del queries
                        del query_index
                        queries = []
                        query_index = []

    if queries:
        save_queries(queries, query_index)
        del queries
        del query_index

    conn.commit()

    print("\nProcessing Responses\n")
    with open(filename_a, newline='', encoding='cp850') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):

            id = str(row['id'])
            comment = str(row['body'])
            thread_id = str(row['link_id'])[3:]
            parent_id = str(row['parent_id'])[3:]

            if len(comment.split(" ")) <= 300:
                if parent_id == thread_id:
                    if parent_id in thread_index:
                        responses.append(comment)
                        response_parent_index.append(parent_id)
                        response_thread_index.append(thread_id)
                        if len(responses) > 1000:
                            print(i)
                            save_responses(
                                responses, response_parent_index, response_thread_index)
                            del responses
                            del response_parent_index
                            del response_thread_index
                            responses = []
                            response_parent_index = []
                            response_thread_index = []

        if responses:
            save_responses(responses, response_parent_index,
                           response_thread_index)
            del responses
            del response_parent_index
            del response_thread_index

    conn.commit()

conn.close()
