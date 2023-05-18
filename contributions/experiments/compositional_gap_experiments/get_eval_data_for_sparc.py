import os, json
import pandas as pd
from execute_query import run_query, convert_stringlist_to_list

eval_path = "bert_run_true_1-step35100_eval.json"
fobj = open(eval_path)
data = json.load(fobj)
items = data['per_item']
fobj.close()

dev_json_file_path = "dev.json"
f = open(dev_json_file_path, "r")
input_json = json.load(f)
n = len(input_json)
interaction_list = []
for i in range(n):
    # ith objective
    sparc_item = input_json[i]
    db_id = sparc_item['database_id']
    list_idx = []
    for interaction in sparc_item['interaction']:
        interaction_list.append(interaction['utterance'])

df_sel_qns = pd.read_csv("final_set_questions_with_results_and_predictions.csv")
db_list = list(df_sel_qns['database_id'])
utterance_list = list(df_sel_qns['utterance'])
query_list = list(df_sel_qns['query'])
utterance_idx = list(df_sel_qns['utterance_idx'])
interaction_idx = list(df_sel_qns['interaction_idx'])
finalqn_results = list(df_sel_qns['finalqn_results'])
predicted_query = list(df_sel_qns['predicted_query'])
predicted_query_res = list(df_sel_qns['predicted_query_res'])

res_dict = {'db_id': [], 
            'utterance':[],
            'interaction': [],
            'int_idx':[],
            'int_pred':[],
            'pred_res':[],
            'int_gold':[],
            'gold_res':[],
            }

for i in range(len(db_list)):
    list_int_idx = interaction_idx[i]
    list_int_idx = convert_stringlist_to_list(list_int_idx)
    db_id, utt_i = db_list[i], utterance_list[i]
    for int_idx in list_int_idx:
        nl = interaction_list[int(int_idx)]
        nl_q = items[int(int_idx)]["predicted"]
        nl_res_q = run_query(db_id,nl_q)
        #list_nl_res_q = convert_stringlist_to_list(nl_res_q)
        if len(nl_res_q)>10:
            nl_res_q = nl_res_q[:10]
        nl_qg = items[int(int_idx)]["gold"]
        nl_res_qg = run_query(db_id,nl_qg)
        #list_nl_res_qg = convert_stringlist_to_list(nl_res_qg)
        if len(nl_res_qg)>10:
            nl_res_qg = nl_res_qg[:10]
        res_dict['db_id'].append(db_id)
        res_dict['utterance'].append(utt_i)
        res_dict['interaction'].append(nl)
        res_dict['int_idx'].append(int_idx)
        res_dict['int_pred'].append(nl_q)
        res_dict['pred_res'].append(nl_res_q)
        res_dict['int_gold'].append(nl_qg)
        res_dict['gold_res'].append(nl_res_qg)

df = pd.DataFrame.from_dict(res_dict)
df.to_csv("sparc_combined_set_questions_with_results.csv")



