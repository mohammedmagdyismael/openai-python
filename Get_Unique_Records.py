""" 
    example:
    $ python Get_Unique_Records.py ./fileA.csv ./fileB.csv common_column_name output.csv
"""

import sys
import pandas as pd

file_A = sys.argv[1]
file_B = sys.argv[2]
common_column = sys.argv[3]
output_file = sys.argv[4]

# Merge the two dataframes on the 'key' column
merged = pd.merge(file_B, file_A, on=common_column, how='left', indicator=True)

# Select only the rows where the right dataframe is null
unique_to_target_file_df = merged[merged['_merge'] == 'left_only']

# Save the unique rows to a new CSV file
unique_to_target_file_df.to_csv(output_file, index=False)
