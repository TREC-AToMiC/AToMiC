# Experiment Notebook for BM25 baseline

## Requirements

- [Anserini](https://github.com/castorini/anserini): Index and search
- [trec_eval](https://github.com/usnistgov/trec_eval): Evaluation
- [textacy](https://textacy.readthedocs.io/en/latest): Simple preprocessing

## Guides

1. Prepare Anserini/trec_eval and modify the corresponding path in `index_anserini.sh` and `search_anserini.sh`
2. Prepare qrel (text format) by:
```python
from datasets import load_dataset

SPLIT = "<split>" # train, validation, test
T2I_FILE = "<filepath>"
I2T_FILE = "<filepath>"

qrel_ds = load_dataset('TREC-AToMiC/AToMiC-Qrels-v0.2')
qrel_ds[SPLIT].to_csv(T2I_FILE, header=None, sep=' ', index=False) # T2I qrel
qrel_ds[SPLIT].to_csv(T2I_FILE, colmns=['image_id', 'Q0', 'text_id', 'rel'), header=None, sep=' ', index=False) # I2T qrel
```
3. run `index_anserini.sh` and `search_anserini.sh`
