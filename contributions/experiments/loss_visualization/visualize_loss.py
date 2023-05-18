import os
import pandas as pd
from viz_helper import generate_viz

truncate_step = None
loss_logs_dir = os.getcwd()
loss_dict = {
    #'wikisql_glove': {'overall': {'step':[], 'value':[]}, 'train': {'step':[], 'value':[]}, 'val': {'step':[], 'value':[]}},
    #'spider_glove': {'overall': {'step':[], 'value':[]}, 'train': {'step':[], 'value':[]}, 'val': {'step':[], 'value':[]}},
    #'spider_bert': {'overall': {'step':[], 'value':[]}, 'train': {'step':[], 'value':[]}, 'val': {'step':[], 'value':[]}},
    'spider_roberta': {'overall': {'step':[], 'value':[]}, 'train': {'step':[], 'value':[]}, 'val': {'step':[], 'value':[]}},
    'spider_gap': {'overall': {'step':[], 'value':[]}, 'train': {'step':[], 'value':[]}, 'val': {'step':[], 'value':[]}},
    'spider_grappa_ssp': {'overall': {'step':[], 'value':[]}, 'train': {'step':[], 'value':[]}, 'val': {'step':[], 'value':[]}},
    'spider_grappa_mlpm_ssp': {'overall': {'step':[], 'value':[]}, 'train': {'step':[], 'value':[]}, 'val': {'step':[], 'value':[]}},  
    }
start, stop = 10000, 40000
for model_name in list(loss_dict.keys()):
    prev_step = {'overall': -1, 'train':-1, 'val':-1}
    file_name = f"log_{model_name}.txt"
    file_path = os.path.join(loss_logs_dir, "logs",file_name)
    if not os.path.exists(file_path):
        continue
    with open(file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Logging" in line:
                continue
            new_line = line.split("Step ")[1]
            comps = new_line.split(" ")
            if comps[1] == "stats,":
                step = int(comps[0])
                mode = comps[2][:-1]
                if step%100==0 and step>prev_step[mode] and step>=start and step<=stop:
                    val = float(comps[-1][:-2])
                    loss_dict[model_name][mode]['step'].append(step)
                    loss_dict[model_name][mode]['value'].append(val) 
                    prev_step[mode] = step
            else:
                step = int(comps[0][:-1])
                if step%100==0 and step>prev_step["overall"] and step>=start and step<=stop:
                    val = float(comps[-1].split("=")[1][:-2])
                    loss_dict[model_name]["overall"]['step'].append(step)
                    loss_dict[model_name]["overall"]['value'].append(val)
                    prev_step["overall"] = step

df_dict = {'overall': None, 'train':None, 'val':None}
for mode in list(df_dict.keys()):
    df_temp = pd.DataFrame()
    for model_name in list(loss_dict.keys()):
        df_model = pd.DataFrame.from_dict(loss_dict[model_name][mode])
        df_model['model_name'] = model_name
        df_temp = pd.concat([df_temp, df_model], axis=0)
    df_dict[mode] = df_temp
generate_viz(df_dict['train'], "train",start, stop)
generate_viz(df_dict['val'], "val",start, stop)
generate_viz(df_dict['overall'], "overall",start, stop)
# df_dict['train'].to_csv("train_loss.csv")
# df_dict['val'].to_csv("val_loss.csv")
# df_dict['loss'].to_csv("overall_loss.csv")
