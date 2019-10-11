from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
import numpy as np
from konlpy.tag import Mecab
mecab = Mecab()

class NLPcore():
    def __init__(self, TH, query):
        self.query = query
        self.TH = TH
        self.cap_path = datapath("/home/ubuntu/seungho/fastText/build/run11_chat_mecab_190824.bin")
        self.model = load_facebook_vectors(self.cap_path)
        self.example = self.model['안녕']
        
        
    def get_sentence_vec(self, A):
        res = mecab.morphs(A)
        vec = np.zeros_like(self.example)
        for morph in res:
            vec += self.model[morph]
        return vec
    
    
    def queryDB(self, DB):
        """
        query must be a string
        DB must be numpy array of string
        
        returns a list of targets
        if target is None, empty list is returned
        """
        DB_vec_list = list(map(self.get_sentence_vec, DB.tolist()))

        A = np.array(DB_vec_list)
        B = np.array(self.get_sentence_vec(self.query)).reshape(-1, 1)
        inner = np.matmul(A, B)

        NA = np.linalg.norm(A, axis=1, keepdims=True)
        NB = np.linalg.norm(B, axis=0, keepdims=True)
        norm = np.matmul(NA, NB)
        sim = inner / norm
        target_idx = sim > self.TH
        print('target counts:', np.sum(target_idx))
        print('targets:', DB[target_idx])
        return DB[target_idx].tolist()