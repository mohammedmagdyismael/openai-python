""" 
    example:
    $ python Generate_Embeddings.py ./data_source_path.csv column_name_to_embed embeddings_output_file.csv
"""

import sys
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding
from openai_apikey import openai_api_key


openai.api_key = openai_api_key
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"

data_source_file = sys.argv[1]
name_col_to_Embed = sys.argv[2]
outut_file_path = sys.argv[3]

print('Generating Embeddings File In Progress ....')
input_datapath = data_source_file
df = pd.read_csv(input_datapath)
df = df.dropna()
encoding = tiktoken.get_encoding(embedding_encoding)
df["n_tokens"] = df[name_col_to_Embed].apply(lambda x: len(encoding.encode(x)))
df["embedding"] = df[name_col_to_Embed].apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv(outut_file_path)
print('Generating Embeddings File Is Done !')