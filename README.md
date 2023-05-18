# Learning Schematic and Contextual Representations for Text-to-SQL Parsing

## What each file does ?
### Folders
- configs: this folder consists of base configuration files required for ratsql which sets the model parameters [1]
- contributions: this folder consists of our experiments
- experiments: configuration files required for different experiments for WikiSQL+glove, Spider+glove, Spider+BER [1]
- gap: consists of necessary codes required for running GAP model [2]
- grappa: follows the same structure for running GraPPA model [3]
- roberta: follows the same structure required for running RoBERTa model [1][4]
- ratsql: follows the same structure required for running WikiSQL+glove, Spider+glove, Spider+BERT model [1]

Common structure followed throughout for the root folder, gap,  and roberta is as follows

For grappa, the structure followed is as below:



## Which files were writting by your group ?
- contributions/error_analysis consists of the file that automates the error analysis and annotations for few samples.
- contributions/loss_visualization consists of files to help visualize loss plots


## Which files were tweaked/configured by your group ?
- we updated the Dockerfile to make the ratsql docker instance run

## References:
1. https://github.com/microsoft/rat-sql
2. https://github.com/awslabs/gap-text2sql/
3. ...
4. https://github.com/ReinierKoops/rat-sql.git
5. https://github.com/taoyds/spider