# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
# import stanza
import sqlite3
import json
import os
import _jsonnet
from seq2struct.commands.infer import Inferer
from seq2struct.datasets.spider import SpiderItem
from seq2struct.utils import registry
import torch
from nltk.tokenize import word_tokenize
import pandas as pd

def check_datatype(val):
    # if isinstance(val,tuple) or isinstance(val,list):
    #     if len(val)>1:
    #         return val
    #     else:
    #         return val[0]    
    # else:
    #     return val
    return val

def run_query(db_id, query):
    try:         
        root_path = os.getcwd()
        data_dir = os.path.join(root_path, "data","spider-bart", "database",f"{db_id}")
        db_path = os.path.join(data_dir,f'{db_id}.sqlite')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute an SQL query
        cursor.execute(query)

        # Fetch the results
        results = cursor.fetchall()
        res = []
        # Process the results
        for row in results:
            res.append(check_datatype(row))
        # Close the database connection
        conn.close()
        return res
    except:
        return []

root_dir = os.getcwd()
exp_config_path = os.path.join(root_dir,"experiments", "spider-configs", "gap-run.jsonnet")
model_dir = os.path.join(root_dir,"logdir","bart_run_1","bs=12,lr=1.0e-04,bert_lr=1.0e-05,end_lr=0e0,att=1")
print("model_dir={}".format(model_dir))
print("check if path exists = {}".format(os.path.exists(model_dir)))
checkpoint_step = 41000

exp_config = json.loads(_jsonnet.evaluate_file(exp_config_path))
model_config_path = os.path.join(root_dir, exp_config["model_config"])
model_config_args = exp_config.get("model_config_args")
infer_config = json.loads(_jsonnet.evaluate_file(model_config_path, tla_codes={'args': json.dumps(model_config_args)}))
print(" ### INFER CONFIG ###")
print(infer_config)

inferer = Inferer(infer_config)
inferer.device = torch.device("cpu")
model = inferer.load_model(model_dir, checkpoint_step)
dataset = registry.construct('dataset', inferer.config['data']['val'])
print(" ######### DATASET #######")
print(dataset)

for id, schema in dataset.schemas.items():
    model.preproc.enc_preproc._preprocess_schema(schema)

def question(q, db_id):
    spider_schema = dataset.schemas[db_id]
    colMap = {}
    data_item = SpiderItem(
        text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
        code=None,
        schema=spider_schema,
        orig_schema=spider_schema.orig,
        orig={"question": q}
    )
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None
    with torch.no_grad():
        m2c_align_mat, m2t_align_mat = None, None
        res, m2c_align_mat, m2t_align_mat  = inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)
        #res = inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)
        #return res, m2c_align_mat, m2t_align_mat, spider_schema, data_item
        return res

df_path = os.path.join(os.getcwd(),"final_set_questions_with_results.csv")
final_df = pd.read_csv(df_path)

db_list = list(final_df['database_id'])
utterance_list = list(final_df['utterance'])
query_list = list(final_df['query'])
utterance_idx = list(final_df['utterance_idx'])
interaction_idx = list(final_df['interaction_idx'])
finalqn_results = list(final_df['utterance_result'])

predicted_query = []
predicted_query_res = []
for utterance,query,db_id in zip(utterance_list, query_list, db_list):
  res = question(utterance, db_id)
  pred_query = res[0]['inferred_code']
  pred_query_res = run_query(db_id,pred_query)
  print(pred_query, pred_query_res)
  predicted_query.append(pred_query)
  predicted_query_res.append(pred_query_res)

final_ques_df_dict = {'database_id':db_list,
                 'utterance': utterance_list,
                 'query': query_list,
                 'utterance_idx': utterance_idx,
                 'interaction_idx': interaction_idx,
                 'finalqn_results': finalqn_results,
                 'predicted_query': predicted_query,
                 'predicted_query_res': predicted_query_res
                 }

final_ques_df = pd.DataFrame.from_dict(final_ques_df_dict)
op_path = os.path.join(os.getcwd(),"final_set_questions_with_results_and_predictions.csv")
final_ques_df.to_csv(op_path)
# ques, db_id = "For the cars with 4 cylinders, which model has the largest horsepower?", "car_1"
# res, m2c_align_mat, m2t_align_mat, spider_schema, data_item = question(ques, db_id)
# #res = question(ques, db_id)
# print("")
# print("")
# print(f"For natural language question: {ques} (asked in db_id {db_id})")
# print("")
# print(f"result code: {res}")
# print("")
# print(f"m2c_align_mat.size()={m2c_align_mat.size()}")
# print("")
# print(f"m2t_align_mat.size()={m2t_align_mat.size()}")

# #print(m2c_align_mat.size())
# #print(m2t_align_mat.size())

# align_mat = torch.cat((m2c_align_mat, m2t_align_mat),1)
# fig, ax = plt.subplots()
# # Set the colormap
# cmap = plt.cm.Reds
# # Create a heatmap
# heatmap = ax.pcolor(align_mat, cmap=cmap)
# plt.colorbar(heatmap)

# def generate_alignment_matrix_viz_spider(m2c_align_mat, m2t_align_mat, nl_tokens, schema_list):
#     align_mat = torch.cat((m2c_align_mat, m2t_align_mat),1)
#     n, m = len(nl_tokens), len(schema_list)
#     align_mat = align_mat[:n,:]
#     align_mat = align_mat.cpu().detach().numpy()
#     # Create a figure and axis object
#     fig, ax = plt.subplots(figsize=(30, 20))
#     fig, ax = plt.subplots()

#     # Set the colormap
#     cmap = plt.cm.Reds
#     # Create a heatmap
#     heatmap = ax.pcolor(align_mat, cmap=cmap)
#     # Set the ticks and labels
#     # reverse the order of yaxis labels
#     ytick_labels = nl_tokens
#     xtick_labels = schema_list
#     yticks = np.arange(0.5, len(ytick_labels), 1)
#     xticks = np.arange(0.5, len(xtick_labels), 1)
    
#     ax.set_xticks(xticks, minor=False)
#     ax.set_xticklabels(xtick_labels, minor=False, rotation=45, ha='right')
#     ax.set_yticks(yticks, minor=False)
#     ax.set_yticklabels(ytick_labels, minor=False)
    
#     # Add the colorbar
#     plt.colorbar(heatmap)
#     # Set the title and axis labels
#     plt.title('Alignment Matrix')
#     plt.xlabel('Columns')
#     plt.ylabel('Tokens')
#     plt.savefig("alignmentGAP.png")
#     plt.show()

# list_tables = []
# list_cols = ["column:*"]
# for table in spider_schema.tables:
#     table_name = table.orig_name
#     list_tables.append(table_name)
#     for col in table.columns:
#         list_cols.append(f"column:{table_name}.{col.orig_name}")
# print(list_cols + list_tables)
# nl_tokens = word_tokenize(ques)
# # tokenizer_ques = stanza.Pipeline(lang="en", processors="tokenize")
# # doc = tokenizer_ques(ques)
# # nl_tokens = []
# # print(type(doc.sentences[0]))
# # for token_kv in doc.sentences[0].tokens:
# #     nl_tokens.append(token_kv.text)
# #     print(nl_tokens)

# schema_list = list_cols + list_tables
# generate_alignment_matrix_viz_spider(m2c_align_mat, m2t_align_mat, nl_tokens, schema_list)