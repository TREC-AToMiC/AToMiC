# Experiment Notebook for BM25 baseline

## Requirements

- [Pyserini](https://github.com/castorini/pyserini): Index and search
- [textacy](https://textacy.readthedocs.io/en/latest): Simple preprocessing
- [datasets](https://huggingface.co/docs/datasets/index): Dataset access

## Guides

1. Install requirements: `pip install -r requirements.txt`
2. Run the script to generate the run files: `python run_bm25_baselines.py`
3. Evaluate using the following commands, replacing `{SPLIT}` and `{SETTING}` appropriately:
- Text to Image
```
python -m pyserini.eval.trec_eval -c -m recip_rank -M 10 qrels/qrels.atomic.{SPLIT}.t2i.trec runs/run.{SPLIT}.bm25-anserini-default.t2i.{SETTING}.trec
python -m pyserini.eval.trec_eval -c -m recall.10,1000 qrels/qrels.atomic.{SPLIT}.t2i.trec runs/run.{SPLIT}.bm25-anserini-default.t2i.{SETTING}.trec
```
- Image to Text
```
python -m pyserini.eval.trec_eval -c -m recip_rank -M 10 qrels/qrels.atomic.{SPLIT}.i2t.trec runs/run.{SPLIT}.bm25-anserini-default.i2t.{SETTING}.trec
python -m pyserini.eval.trec_eval -c -m recall.10,1000 qrels/qrels.atomic.{SPLIT}.i2t.trec runs/run.{SPLIT}.bm25-anserini-default.i2t.{SETTING}.trec
```

## Reproduction Log
+ Results reproduced by [@dlrudwo1269](https://github.com/dlrudwo1269) on <2023-04-09> (commit [`45a5540`](https://github.com/dlrudwo1269/AToMiC/commit/45a5540c473c48e4e7c68b0258b5ad23cf2e43d0))
