""" 
    example:
    $ python Merge_CSV.py ./fileA.csv ./fileB.csv "common_col_A, common_col_B, common_col_C" output.csv
"""

import sys
import pandas as pd
import ast
from openai_apikey import openai_api_key

file_A = sys.argv[1]
file_B = sys.argv[2]
common_columns_list = sys.argv[3]
output_file = sys.argv[4]

# Merge the two dataframes on the 'key' column
merged = pd.merge(file_B, file_A, on=common_columns_list.split(','), how='left', indicator=True)

# Select only the rows where the right dataframe is null
unique_to_target_file_df = merged[merged['_merge'] == 'left_only']

# Save the unique rows to a new CSV file
unique_to_target_file_df.to_csv(output_file, index=False)
