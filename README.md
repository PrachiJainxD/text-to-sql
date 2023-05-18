# Learning Schematic and Contextual Representations for Text-to-SQL Parsing

### Folders
- configs: this folder consists of base configuration files required for ratsql which sets the model parameters [1]
- contributions: this folder consists of our experiments
- experiments: configuration files required for different experiments for WikiSQL+glove, Spider+glove, Spider+BER [1]
- gap: consists of necessary codes required for running GAP model [2]
- grappa: follows the same structure for running GraPPA model [3]
- roberta: follows the same structure required for running RoBERTa model [1][4]
- ratsql: follows the same structure required for running WikiSQL+glove, Spider+glove, Spider+BERT model [1]

Above common structure is followed for ratsql, gap/rat-sql-gap, grappa/spider, grappa/spider_ssp

## Which files were written by your group ?
Under contributions
- error_analysis consists of the file that automates and annotations for error analysis
- EDA/ consists of the python script to generate stats mentioned under datasets section
- training/ consists of the colab notebook setup for training RoBERTa
- Evaluation/ consists of data derived of running evaluation
- experiments consists of files to run experiments with respect to question/subquestion answering, loss visualization and paraphrasing or rewritten utterances
- inference - consists of inference scripts written for different models
- sparc_preprocessing - consists of the script needed for preprocesssing sparc to spider equivalent
- environment_setup_helper - consists of code for setting up environment on colab
- Results - consists of results from running evaluation metrics


## References:
1. https://github.com/microsoft/rat-sql
2. https://github.com/awslabs/gap-text2sql/
3. https://github.com/taoyds/grappa/tree/main/grappa
4. https://github.com/ReinierKoops/rat-sql.git
5. https://github.com/taoyds/spider