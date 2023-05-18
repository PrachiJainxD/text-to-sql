import json
import os
import _jsonnet
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem
from ratsql.utils import registry
import torch

exp_config_path = "/app/experiments/wikisql-glove-run.jsonnet"
root_dir = "/app/"
model_dir = "/app/logdir/glove_run"
checkpoint_step = 30100

exp_config = json.loads(_jsonnet.evaluate_file(exp_config_path))
model_config_path = os.path.join(root_dir, exp_config["model_config"])
model_config_args = exp_config.get("model_config_args")
infer_config = json.loads(_jsonnet.evaluate_file(model_config_path, tla_codes={'args': json.dumps(model_config_args)}))

inferer = Inferer(infer_config)
inferer.device = torch.device("cpu")
model = inferer.load_model(model_dir, checkpoint_step)
dataset = registry.construct('dataset', inferer.config['data']['val'])

#for _, schema in dataset.schemas.items():
for id, schema in dataset.schema_dicts.items():
    model.preproc.enc_preproc._preprocess_schema(schema)

def convert_to_sql(wikisql_q, col_map, tb_id):
    '''
    e.g. input:
    [
        {
            'orig_question': 'What position does the player who played for butler play?', 
            'model_output': {'_type': 'select', 'agg': {'_type': 'NoAgg'}, 'col': 3, 'conds': [{'_type': 'cond', 'op': {'_type': 'Equal'}, 'col': 0, 'value': 'butler'}]}, 
            'inferred_code': {
                'agg': 0, 
                'sel': 3, 
                'conds': [[0, 0, 'butler']]
                }, 
                'score': -0.6181345462000536
        }
    ]
    - `sql`: the SQL query corresponding to the question. This has the following subfields:
        - `sel`: the numerical index of the column that is being selected. You can find the actual column from the table.
        - `agg`: the numerical index of the aggregation operator that is being used. You can find the actual operator from `Query.agg_ops` in `lib/query.py`.
        - `conds`: a list of triplets `(column_index, operator_index, condition)` where:
            - `column_index`: the numerical index of the condition column that is being used. You can find the actual column from the table.
            - `operator_index`: the numerical index of the condition operator that is being used. You can find the actual operator from `Query.cond_ops` in `lib/query.py`.
            - `condition`: the comparison value for the condition, in either `string` or `float` type.
    '''
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    wikisql_gen_query = wikisql_q[0]['inferred_code']
    # 1. constructing the select clause
    if wikisql_gen_query['agg'] == 0: # no aggregation in sql query
        sql_query = f"SELECT {col_map[wikisql_gen_query['sel']]} "
    else: # aggregation based on agg_ops
        sql_query = f"SELECT {agg_ops[wikisql_gen_query['agg']]}({col_map[wikisql_gen_query['sel']]}) "
    
    # 2. constructing from clause
    sql_query = sql_query + f"FROM {tb_id}"
    for cond_list in wikisql_gen_query['conds']:
        col_idx, op_idx, condition = cond_list[0], cond_list[1], cond_list[2]
        where_condn = f" WHERE {col_map[col_idx]} {cond_ops[op_idx]} {condition}"
        sql_query = sql_query + where_condn

    return wikisql_q, wikisql_gen_query, sql_query

def question(q, db_id):
    #spider_schema = dataset.schemas[db_id]
    spider_schema = dataset.schema_dicts[db_id]
    colMap = {}
    for table_obj in spider_schema.tables: # spider_schema.tables => tuple based on ratsql/datasets/spider.py
        for col_obj in table_obj.columns:
            colMap[col_obj.id]= col_obj.orig_name
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
    res = ""
    with torch.no_grad():
        res= inferer._infer_one(model, data_item, preproc_data, beam_size=1, output_history=True, use_heuristic=False)

    return convert_to_sql(res, col_map=colMap,tb_id=db_id)

#ques = "What position does the player who played for butler play?"
ques = "How many different college/junior/club teams provided a player to the Washington Capitals NHL Team?"
#tbl_id = "1-10015132-11"
tbl_id = "1-1013129-2"
wikisql_q, wikisql_inferred_code, sql_query = question(ques, tbl_id)
print("")
print("")
print(f"For natural language question: {ques} in table_id {tbl_id}")
print("")
print(f"actual output from code: {wikisql_q}")
print("")
print(f"wikisql inferred code: {wikisql_inferred_code}")
print("")
print(f"generated sql query: {sql_query}")
print("")