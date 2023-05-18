import json, os
# import nltk
# nltk.download('punkt')
import pandas as pd
from error_analysis import build_foreign_key_map_from_json, evaluate
def execution_accuracy_metrics(exp_results, gold_query_sql_path, all_tables_json, all_db_path, nl_qns, etype="all"):
    predict = []
    # for k in range(len(exp_results)):
    #     predict.append('SELECT'+exp_results[k][0]['text'].replace(';','').replace('\n',''))
    for val in range(len(exp_results['per_item'])):
        predict.append(exp_results['per_item'][val]['predicted'])
    assert etype in ["all", "exec", "match"], "Unknown evaluation method"

    kmaps = build_foreign_key_map_from_json(all_tables_json)
    results = evaluate(gold_query_sql_path, predict, all_db_path, etype, kmaps, nl_qns)
    return results

data_dir = os.path.join(os.getcwd(),"spider")
eval_file_path = os.path.join(os.getcwd(), "bert_run_true_1-step34100_eval.json")
#infer_file_path = os.path.join(os.getcwd(), "bert_run_true_1-step10300_infer.json")

table_path = os.path.join(data_dir,"tables.json") # 'tables.json'
dev_gold_path = os.path.join(data_dir,"dev_gold.sql") #'dev_gold.sql'
database_path = os.path.join(data_dir,"database")
f = open(eval_file_path, "r")
eval_res = json.load(f)

file_path = os.path.join(os.getcwd(),"bert_run_true_1-step34100.infer")
nl_qns = []
with open(file_path,'r') as f:
    lines = f.readlines()
    for line in lines:
        res = json.loads(line)
        nl_qns.append(res['beams'][0]['orig_question'])

res_val = execution_accuracy_metrics(eval_res, dev_gold_path, table_path, database_path,nl_qns)
pd.DataFrame(res_val).to_csv("generate_data_spider_roberta.csv")