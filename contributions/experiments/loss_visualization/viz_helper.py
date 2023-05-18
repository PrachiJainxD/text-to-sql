from matplotlib import pyplot as plt
label_dict = {
    'wikisql_glove': "WikiSQL+GloVe",
    'spider_glove': "Spider+GloVe",
    'spider_bert': "Spider+BERT",
    'spider_roberta': "Spider+RoBERTa",
    'spider_gap':"Spider+GAP",
    'spider_grappa_ssp':"Spider+GraPPa (MLM+SSP)",
    'spider_grappa_mlpm_ssp':"Spider+GraPPa (SSP)"
}

def generate_viz(df, mode, start, stop):
    '''
    To change the order of legends:
    https://www.statology.org/matplotlib-legend-order/
    https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined
    '''
    model_names = set(list(df['model_name']))
    
    #print(len(xvals))
    fig, ax = plt.subplots()
    for model_name in model_names:
        subset_df = df[df['model_name']==model_name]
        print(subset_df.shape)
        if subset_df.shape[0]==0:
            continue
        x_val = list(subset_df['step'])
        y_val = list(subset_df['value'])
        ax.plot(x_val, y_val, label=label_dict[model_name])
    
    ax.set_xlabel("Steps")
    ax.set_ylabel(f"Loss")
    ax.set_title(f"Plot for {mode} loss")
    #ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend()
    plt.savefig(f"{mode}_{start}_{stop}.png")