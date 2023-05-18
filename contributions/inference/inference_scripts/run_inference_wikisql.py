import json
import os
import _jsonnet
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem
from ratsql.utils import registry
import torch
from inference_helper import GloVe, generate_alignment_matrix_viz

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

def generate_tokens_from_nl(nl_ques):
    gloveObj = GloVe("/mnt/data/", kind="42B")
    list_sent_tokens = gloveObj.tokenize(nl_ques)
    return list_sent_tokens

def generate_embeddings(nl_ques, list_cols, tbl_id):
    gloveObj = GloVe("/mnt/data/", kind="42B")
    list_sent_tokens = gloveObj.tokenize(nl_ques)
    list_schema_tokens = []
    for col in list_cols:
        t_list = col.split(" ")
        t_list = [t.lower() for t in t_list]
        list_schema_tokens = list_schema_tokens + t_list
    list_all_schema_tokens = list_schema_tokens + [tbl_id]
    list_tokens = list(set(list_sent_tokens + list_all_schema_tokens))
    store_embeddings = {}
    for token in list_tokens:
        tokenEmb = gloveObj.lookup(token)

        if isinstance(tokenEmb, type(None)):
            print(f"{token} not present in vocab")
            store_embeddings[token] = None
        else:
            print(f"{token} with size = {tokenEmb.size()}")
            store_embeddings[token] = tokenEmb
    n,m = len(list_sent_tokens), len(list_schema_tokens)
    alignment_matrix = [[0]*m for _ in range(n)]
    cosine_sim = torch.nn.CosineSimilarity(dim=0)
    for i in range(n):
        sent_token = list_cols[i]
        if sent_token not in store_embeddings:
            alignment_matrix[i] = [0]*m
            continue
        for j in range(m):
            schema_token = list_cols[j]
            if schema_token in store_embeddings:
                alignment_matrix[i][j] = cosine_sim(store_embeddings[sent_token], store_embeddings[schema_token])
            else:
                alignment_matrix[i][j] = 0
    x_labels = [f"column:{col}" for col in list_cols] + [f"table:{tbl_id}"]
    y_labels = list_sent_tokens
    return alignment_matrix, x_labels, y_labels
    

    
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

    return [wikisql_q, wikisql_gen_query, sql_query, list(col_map.values())]

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
    # enc_input is a dictionary with keys such as ['raw_question', 'question', 'question_for_copying', 
    # 'db_id', 'sc_link', 'cv_link', 'columns', 'tables', 'table_bounds', 'column_to_table', 'table_to_columns', 
    # 'foreign_keys', 'foreign_keys_tables', 'primary_keys'])
    preproc_data = enc_input, None
    #print(type(enc_input)) # dict
    #print(enc_input.keys())
    #enc_input['sc_link'].keys() => ['q_col_match', 'q_tab_match']
    #print(enc_input['sc_link']['q_col_match']) # empty dict
    #print(enc_input['sc_link']['q_tab_match']) # empty dict
    #print(enc_input['cv_link'].keys()) # ['num_date_match', 'cell_match']
    #_ = input("enter ?")
    res = ""
    m2c_align_mat, m2t_align_mat = None, None
    with torch.no_grad():
        res, m2c_align_mat, m2t_align_mat= inferer._infer_one(model, data_item, preproc_data, beam_size=1, output_history=True, use_heuristic=False)
    return convert_to_sql(res, col_map=colMap,tb_id=db_id), m2c_align_mat, m2t_align_mat

ques, tbl_id = "Name the least game for january 29", "1-23286112-8"
ret_list, m2c_align_mat, m2t_align_mat = question(ques, tbl_id)
wikisql_q, wikisql_inferred_code, sql_query, list_cols = ret_list[0],ret_list[1],ret_list[2],ret_list[3]
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
print(f"m2c_align_mat.size()={m2c_align_mat.size()}")
print("")
print(f"m2t_align_mat.size()={m2t_align_mat.size()}")
nl_tokens = generate_tokens_from_nl(ques)
print(f"nl_tokens = {nl_tokens}")
print(list_cols)
print(tbl_id)
generate_alignment_matrix_viz(m2c_align_mat,m2t_align_mat, nl_tokens, list_cols, tbl_id)
# print("Invoking the generate embeddings function")
# generate_embeddings(ques, list_cols, tbl_id)