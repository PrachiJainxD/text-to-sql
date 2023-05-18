import json
import os
import _jsonnet
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem
from ratsql.utils import registry
import torch
from inference_helper import GloVe, generate_alignment_matrix_viz

exp_config_path = "/app/experiments/spider-roberta-run.jsonnet"
root_dir = "/app/"
model_dir = os.path.join("/app/logdir","roberta_run","roberta_bs=4,lr=1.0e-04,bert_lr=1.0e-05,end_lr=0e0,att=1")
print("model_dir={}".format(model_dir))
print("check if path exists = {}".format(os.path.exists(model_dir)))
checkpoint_step = 33100

exp_config = json.loads(_jsonnet.evaluate_file(exp_config_path))
model_config_path = os.path.join(root_dir, exp_config["model_config"])
model_config_args = exp_config.get("model_config_args")
infer_config = json.loads(_jsonnet.evaluate_file(model_config_path, tla_codes={'args': json.dumps(model_config_args)}))
print(" ### INFER CONFIG ###")
print(infer_config)
#_ = input("continue ?")

inferer = Inferer(infer_config)
inferer.device = torch.device("cpu")
model = inferer.load_model(model_dir, checkpoint_step)
dataset = registry.construct('dataset', inferer.config['data']['val'])
print(" ######### DATASET #######")
print(dataset)
_ = input("continue ?")

for id, schema in dataset.schemas.items():
    model.preproc.enc_preproc._preprocess_schema(schema)


def question(q, db_id):
    spider_schema = dataset.schemas[db_id]
    colMap = {}
    # for table_obj in spider_schema.tables: # spider_schema.tables => tuple based on ratsql/datasets/spider.py
    #     #colMap[table_obj.name] = {'tbl_id': table_obj.id, 'columns':{}}
    #     print(table_obj)
    #     print(dir(table_obj))
    #     print(table_obj.__dict__)
    #     _ = input("continue?")
    #     for col_obj in table_obj.columns:
    #         print(col_obj)
    #         print(dir(col_obj))
    #         print(col_obj.__dict__)
    #         #colMap[table_obj.name]['columns'][col_obj.id]= col_obj.orig_name
    # #print(colMap)
    # _ = input("continue?")
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
        res  = inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)
        return res, m2c_align_mat, m2t_align_mat

ques, db_id = "For the cars with 4 cylinders, which model has the largest horsepower?", "car_1"
res, m2c_align_mat, m2t_align_mat = question(ques, db_id)
print("")
print("")
print(f"For natural language question: {ques} (asked in db_id {db_id})")
print("")
print(f"result code: {res}")
print("")
# print(f"m2c_align_mat.size()={m2c_align_mat.size()}")
# print("")
# print(f"m2t_align_mat.size()={m2t_align_mat.size()}")