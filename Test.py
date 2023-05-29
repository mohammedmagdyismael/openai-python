import sys
import pandas as pd
import numpy as np
import openai
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity


print(sys.argv[1])