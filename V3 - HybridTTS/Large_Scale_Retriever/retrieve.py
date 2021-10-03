import numpy as np
import sys
import pickle
import Utils.functions as utils
import copy
import random
import faiss


def top_candidates(candidates, scores, top=1):
    sorted_score_index = np.flip(np.argsort(scores), axis=-1)
    candidates = [candidates[i] for i in sorted_score_index.tolist()]
    scores = [scores[i] for i in sorted_score_index.tolist()]

    return candidates[0:top], scores[0:top], sorted_score_index.tolist()


def retrive(conn, c, idx, index, query_encoding, query_context, top=5):
    _, I = index.search(query_encoding, k=top)

    idx = [idx[i] for i in I[0].tolist()]

    thread_index = ()
    string_sql = 'SELECT * FROM responses WHERE '

    for i, id in enumerate(idx):
        if i == 0:
            string_sql += 'parent_id = ?'
        else:
            string_sql += ' or parent_id = ?'

        thread_index += (id,)

    candidates = []

    for row in c.execute(string_sql, thread_index):
        comment = str(row[-1])
        candidates.append(comment)

    return candidates
