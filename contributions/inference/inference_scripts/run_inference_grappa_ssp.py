# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt

import numpy as np

import torch

import stanza
import json
import os
import _jsonnet
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem
from ratsql.utils import registry
import torch

root_dir = os.getcwd()
exp_config_path = os.path.join(root_dir,"experiments", "spider-bert-run.jsonnet")
model_dir = os.path.join(root_dir,"logdir","bert_run","bs=8,lr=1.0e-04,bert_lr=1.0e-05,end_lr=0e0,att=1")
print("model_dir={}".format(model_dir))
print("check if path exists = {}".format(os.path.exists(model_dir)))
checkpoint_step = 40000

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
        return res, m2c_align_mat, m2t_align_mat, spider_schema, data_item
    
ques, db_id = "For the cars with 4 cylinders, which model has the largest horsepower?", "car_1"
res, m2c_align_mat, m2t_align_mat, spider_schema, data_item = question(ques, db_id)
#res = question(ques, db_id)
print("")
print("")
print(f"For natural language question: {ques} (asked in db_id {db_id})")
print("")
print(f"result code: {res}")
print("")
print(f"m2c_align_mat.size()={m2c_align_mat.size()}")
print("")
print(f"m2t_align_mat.size()={m2t_align_mat.size()}")


def generate_alignment_matrix_viz_spider(m2c_align_mat, m2t_align_mat, nl_tokens, schema_list):
    align_mat = torch.cat((m2c_align_mat, m2t_align_mat),1)
    n, m = len(nl_tokens), len(schema_list)
    align_mat = align_mat[:n,:]
    align_mat = align_mat.cpu().detach().numpy()
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(15, 8))
    #fig, ax = plt.subplots()

    # Set the colormap
    cmap = plt.cm.Reds
    # Create a heatmap
    heatmap = ax.pcolor(align_mat, cmap=cmap)

    # Set the ticks and labels
    # reverse the order of yaxis labels
    ytick_labels = nl_tokens
    xtick_labels = schema_list
    yticks = np.arange(0.5, len(ytick_labels), 1)
    xticks = np.arange(0.5, len(xtick_labels), 1)
    
    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels(xtick_labels, minor=False, rotation=45, ha='right')
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(ytick_labels, minor=False)

    # Add the colorbar
    plt.colorbar(heatmap)

    plt.title('Alignment Matrix')
    
    plt.savefig("grappa_alignment_matrix.png")

list_tables = []
list_cols = ["column:*"]
for table in spider_schema.tables:
  table_name = table.orig_name
  list_tables.append(table_name)
  for col in table.columns:
    list_cols.append(f"column:{table_name}.{col.orig_name}")
print(list_cols + list_tables)

tokenizer_ques = stanza.Pipeline(lang="en", processors="tokenize")
doc = tokenizer_ques(ques)
nl_tokens = []
print(type(doc.sentences[0]))
for token_kv in doc.sentences[0].tokens:
  nl_tokens.append(token_kv.text)
print(nl_tokens)

schema_list = list_cols + list_tables
generate_alignment_matrix_viz_spider(m2c_align_mat, m2t_align_mat, nl_tokens, schema_list)
