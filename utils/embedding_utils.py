import os
import numpy as np

def load_embedding(filename):
    """
    load embedding as python dictionary {root<str>: embeddings<np_array>}
    :param filename: embedding.txt
    :return: dictionary object mapping root to embeddings
    """
    if not os.path.exists(filename):
        print("please run 'Stored Pre-Trained Embeddings Cell!'")
    else:
        with open(filename, "r") as f:
            lines = f.readlines()
            f.close()
            # create map of words to vectors
            embedding = dict()
            for line in lines:
                comp = line.split()
                # map of <str, numpy array>
                embedding[comp[0]] = np.asarray(comp[1:], dtype='float32')
            return embedding