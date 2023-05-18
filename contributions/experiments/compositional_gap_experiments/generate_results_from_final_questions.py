import pandas as pd
import os
from execute_query import run_query

csv_file_name = os.path.join(os.getcwd(), "final_set_questions.csv")
final_df = pd.read_csv(csv_file_name)
res_dict = {}
db_list = list(final_df['database_id'])
utterance_list = list(final_df['utterance'])
query_list = list(final_df['query'])
utterance_idx = list(final_df['idx'])
interaction_idx = list(final_df['list_idx'])
res_dict['database_id'] = db_list
res_dict['utterance'] = utterance_list
res_dict['query'] = query_list
res_dict['utterance_idx'] = utterance_idx
res_dict['interaction_idx'] = interaction_idx
res_list = []

n = len(db_list)
for i in range(n):
    res = run_query(db_list[i], query_list[i])
    res_list.append(res)
res_dict['utterance_result']= res_list
df = pd.DataFrame.from_dict(res_dict)
df.to_csv("final_set_questions_with_results.csv")
