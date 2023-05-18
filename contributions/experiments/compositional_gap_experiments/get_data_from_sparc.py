import os, json
import pandas as pd

def convert_to_spider(file_path, folder_path, file_name):
    f = open(file_path, "r")
    input_json = json.load(f)
    n = len(input_json)
    final_dict = {'database_id':[], 'utterance':[], 'query':[], 'list_idx':[]}
    idx = 0
    for i in range(n):
        # ith objective
        sparc_item = input_json[i]
        db_id = sparc_item['database_id']
        list_idx = []
        for _ in sparc_item['interaction']:
            list_idx.append(idx)
            idx+=1

        final_utterance = sparc_item['final']['utterance']
        final_query = sparc_item['final']['query']
        #
        final_dict['database_id'].append(db_id)
        final_dict['utterance'].append(final_utterance)
        final_dict['query'].append(final_query)
        final_dict['list_idx'].append(list_idx)

    target_path = os.path.join(folder_path, file_name)
    df = pd.DataFrame.from_dict(final_dict)
    df.to_csv(target_path)


root_dir = os.getcwd()
convert_to_spider(os.path.join(root_dir,"dev.json"), os.getcwd(), "sparc_dataset_with_final_utterance.csv")