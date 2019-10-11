from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
import numpy as np
from konlpy.tag import Mecab
mecab = Mecab()
import multiprocessing as mp
from itertools import repeat
import time, json

chat_file = '../chatpair/1to100_question_190827_2331_post.txt'

cap_path = datapath("/home/ubuntu/seungho/fastText/build/run11_chat_mecab_190824.bin")
model = load_facebook_vectors(cap_path)
example = model['안녕']

def get_sentence_vec(A):
    res = mecab.morphs(A)
    vec = np.zeros_like(example)
    for morph in res:
        vec += model[morph]
    return vec


def queryDB(query, DB):
    """
    query must be a string
    DB must be numpy array of string

    returns a list of targets
    if target is None, empty list is returned
    """
    DB_vec_list = list(map(get_sentence_vec, DB.tolist()))

    A = np.array(DB_vec_list)
    B = np.array(get_sentence_vec(query)).reshape(-1, 1)
    inner = np.matmul(A, B)

    NA = np.linalg.norm(A, axis=1, keepdims=True)
    NB = np.linalg.norm(B, axis=0, keepdims=True)
    norm = np.matmul(NA, NB)
    sim = np.squeeze(inner / norm)
    target_idx = sim > TH
#     print('target counts:', np.sum(target_idx))
#     print('targets:', DB[target_idx])
    return DB[target_idx].tolist()

with open(chat_file, 'r', encoding='utf-8') as f:
    chats = f.read().splitlines()
    print('chat count:', len(chats))
    chat_count = len(chats) - (len(chats)%10000)
    chats = chats[:chat_count]
    chats = np.array(chats).reshape(-1,10000)
    chats = list(chats)


TH = 0.96


with open('query_list.json', 'r', encoding='utf-8') as f:
    query_list = json.load(f)

if __name__ == "__main__":
    for category in query_list:
        result_merged = list()
        st = time.time()
        for query in category['query_list']:
            with mp.Pool(40) as pool:
                result = pool.starmap(queryDB, zip(repeat(query), chats))
            for i in result:
                result_merged += i
        result_merged = set(result_merged)
        with open('result/result_'+category['name'], 'w', encoding='utf-8') as f:
            for line in result_merged:
                f.write(line+'\n')
        print('time for', category['name'], ':', time.time() - st)