import pandas as pd 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct  #Cosine Similarity
import time
pd.set_option('display.max_colwidth', -1)
import uuid

def ngrams(string, n=3):
    string = (re.sub(r'[,-./]|\sBD',r'', string)).upper()
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M * ntop

    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data, indices, indptr), shape=(M, N))


def get_matches_df(sparse_matrix, A, B, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    id_use = np.empty([nr_matches], dtype=object)

    for index in range(0, nr_matches):
        left_side[index] = A[sparserows[index]]
        right_side[index] = B[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
        id_ = uuid.uuid4()
        id_use[index] = 'id_'+ str(id_.node)

    return pd.DataFrame({'left_side': left_side,
                         'right_side': right_side,
                         'similairity': similairity,
                        'id_matching': id_use})


def match_all (df_1,nama_kolom_1,df_2,nama_kolom_2):
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix_payment = vectorizer.fit_transform(df_1[nama_kolom_1])
    tf_idf_matrix_buyer = vectorizer.transform(df_2[nama_kolom_2])
    
    t1 = time.time()
    matches = awesome_cossim_top(tf_idf_matrix_payment, tf_idf_matrix_buyer.transpose(), 1, 0)
    t = time.time()-t1
    print("SELFTIMED:", t)
    
    matches_df = get_matches_df(matches, df_1[nama_kolom_1], df_2[nama_kolom_2], top=0)
    return (matches_df)


df = pd.read_csv("./../datapoc/jan_2021_match_id.csv")
df.head()

df_nielsen = df[(df['SOURCE'] == 'NIELSEN')
df_nielsen.head()

df_acara=df[(df['SOURCE'] == 'GENSCTV')
df_acara.head()

mat = match_all (df_nielsen,'PRODUCT VERSION',df_acara,'PRODUCT VERSION')
print (mat)

matches_df.sort_values(['similairity'], ascending=False).head(10)