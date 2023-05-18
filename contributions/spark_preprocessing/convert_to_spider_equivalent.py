import os, json
import sqlparse

target_dir = os.path.join(os.getcwd(),"folder_01")
def gen_query_toks(q):
    q_parse = sqlparse.parse(q)[0]
    list_tokens = []
    print(dir(q_parse))
    for token in q_parse.tokens:
        print(str(token))
        print(token)
    print(list_tokens)

# to convert a. dev.json; b. train.json c.train_gold.sql
def convert_to_spider(file_path, folder_path, file_name, capture_query=False):
    f = open(file_path, "r")
    input_json = json.load(f)
    n = len(input_json)
    final_list = []
    list_queries = []
    for i in range(n):
        # ith objective
        sparc_item = input_json[i]
        db_id = sparc_item['database_id']
        sparc_list_dict = []
        for utterance in sparc_item['interaction']:
            temp_dict = {'db_id': db_id}
            temp_dict['query'] = utterance['query']
            temp_dict['question'] = utterance['utterance']
            temp_dict['question_toks'] = utterance['utterance_toks']
            temp_dict['sql'] = utterance['sql']
            # need to create 'query_toks' and 'query_toks_no_value'
            temp_dict['query_toks'] = ""
            temp_dict['query_toks_no_value'] = ""
            sparc_list_dict.append(temp_dict)
            if capture_query:
                list_queries.append(f"{utterance['query']}	{db_id}")
        final_list.extend(sparc_list_dict)
    target_path = os.path.join(folder_path, file_name)
    with open(target_path, "w") as final:
        json.dump(final_list, final)
    if capture_query:
        with open(os.path.join(folder_path, 'dev.sql'), 'w') as f:
            for query in list_queries:
                f.write(f"{query}\n")


convert_to_spider(os.path.join(os.getcwd(),"dev.json"), target_dir, "dev.json", capture_query=True)
