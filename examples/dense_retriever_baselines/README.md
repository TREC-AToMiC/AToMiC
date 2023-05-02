
## Env setup

```bash
conda create -n atomic python=3.8
conda activate atomic
conda install -c conda-forge openjdk=11
pip install pyserini
```

```bash
# CPU Only
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 faiss-cpu cpuonly -c pytorch
```

```bash
# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 faiss-cpu -c pytorch -c nvidia
```

```bash
conda install -c huggingface transformers datasets
```

## Guides

### Encode

```bash
python encode.py \
    --output_dir embeddings \
    --inputs TREC-AToMiC/AToMiC-Images-v0.2 \
    --encode_type image \
    --id_column image_id \
    --split validation

python encode.py \
    --output_dir embeddings \
    --inputs TREC-AToMiC/AToMiC-Texts-v0.2.1 \
    --encode_type text \
    --id_column text_id \
    --split validation
```

```bash
python encode.py \
    --output_dir embeddings \
    --inputs TREC-AToMiC/AToMiC-Images-v0.2 \
    --encode_type image \
    --id_column image_id \
    --split validation \
    --dtype fp16 \
    --device cuda 

python encode.py \
    --output_dir embeddings \
    --inputs TREC-AToMiC/AToMiC-Texts-v0.2.1 \
    --encode_type text \
    --id_column text_id \
    --split validation \
    --dtype fp16 \
    --device cuda 

```

### Index

```bash
python index.py --embedding_dir embeddings/image/validation --index indexes/image.validation
python index.py --embedding_dir embeddings/text/validation --index indexes/text.validation
```

### Search

```bash
python search.py \
    --topics embeddings/text/validation \
    --index indexes/image.validation.faiss.flat \
    --hits 1000 \
    --output runs/run.validation.t2i.small.trec

python search.py \
    --topics embeddings/image/validation \
    --index indexes/text.validation.faiss.flat \
    --hits 1000 \
    --output runs/run.validation.i2t.small.trec
```

### Evaluate (Optional)

```bash
python prepare_qrels.py

python -m pyserini.eval.trec_eval -c -m recip_rank -M 10 qrels/validation.qrels.t2i.projected.trec runs/run.validation.t2i.small.trec
python -m pyserini.eval.trec_eval -c -m recall.10,1000 qrels/validation.qrels.t2i.projected.trec runs/run.validation.t2i.small.trec

Results:
recip_rank              all     0.2720
recall_10               all     0.4470
recall_1000             all     0.9399
```

```bash
python -m pyserini.eval.trec_eval -c -m recip_rank -M 10 qrels/validation.qrels.i2t.projected.trec runs/run.validation.i2t.small.trec
python -m pyserini.eval.trec_eval -c -m recall.10,1000 qrels/validation.qrels.i2t.projected.trec runs/run.validation.i2t.small.trec

Results:
recip_rank              all     0.2501
recall_10               all     0.4058
recall_1000             all     0.9272
```

