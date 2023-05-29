import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import sys
import os
from openai_apikey import openai_api_key

openai.api_key = openai_api_key

outdir = './output'
if not os.path.exists(outdir):
    os.mkdir(outdir)

input_file = sys.argv[1]
filename = input_file.split('/')[-1].split('.')[0]
embeddings_file_path = "https://cdn-prelive.drbridge.org/vezeeta-master-list-embeddings/veez_embeddings.csv"


df_embeddings = ''
df_input_file = ''

try:
    df_embeddings = pd.read_csv(embeddings_file_path)
except:
    df_embeddings = pd.read_csv(embeddings_file_path, encoding='ISO-8859-1')

try:
    df_input_file = pd.read_csv(input_file)
except:
    df_input_file = pd.read_csv(input_file, encoding='ISO-8859-1')


print(">>>> File:" + input_file)

df_embeddings["embedding"] = df_embeddings.embedding.apply(eval).apply(np.array)
df_input_file["KEY"] = pd.Series()
df_input_file["NAMEEN"] = pd.Series()
df_input_file["SIMILARITY"] = pd.Series()


for i in range(df_input_file.shape[0]):
    print("Progress: " + str(round((i / df_input_file.shape[0]) * 100, 2)) + "%")
    product_embedding = get_embedding(
        df_input_file.loc[i, "nameen"],
        engine="text-embedding-ada-002"
    )
    df_embeddings["similarity"] = df_embeddings.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    df_sorted = df_embeddings.sort_values("similarity", ascending=False)

    df_input_file.loc[i, 'KEY'] = df_sorted.iloc[0, df_sorted.columns.get_loc('key')]
    df_input_file.loc[i, 'NAMEEN'] = df_sorted.iloc[0, df_sorted.columns.get_loc('nameen')]
    df_input_file.loc[i, 'SIMILARITY'] = df_sorted.iloc[0, df_sorted.columns.get_loc('similarity')]
    print(">>> " + df_input_file.loc[i, "nameen"] + "-------Similarity Result:-------" + df_sorted.iloc[0, df_sorted.columns.get_loc('nameen')])
    print(">>> Similarity: " + str(df_sorted.iloc[0, df_sorted.columns.get_loc('similarity')]))
    print()

df_input_file.to_csv('./output/^',filename+'_mapped.csv', index=False)
