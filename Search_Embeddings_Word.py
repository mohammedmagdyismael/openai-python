""" 
    example:
    $ python Search_Embeddings_Word.py ./embeddings_data_source_path.csv "word_to_search" number_of_output_results
"""

import pprint
import sys
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from openai_apikey import openai_api_key

openai.api_key = openai_api_key
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"

embeddings_data_source_file = sys.argv[1]
search_word = sys.argv[2]
n_output_results = int(sys.argv[3])

print('Searching In Embeddings File In Progress ....')
df = pd.read_csv(embeddings_data_source_file)
df["embedding"] = df.embedding.apply(eval).apply(np.array)
embedding = get_embedding(
    search_word,
    engine="text-embedding-ada-002"
)
df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))
df_sorted = df.sort_values("similarity", ascending=False).head(n_output_results)
pprint.pprint(df_sorted)
