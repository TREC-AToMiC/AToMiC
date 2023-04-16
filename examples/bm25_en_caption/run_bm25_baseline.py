#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pathlib import Path
import argparse

from datasets import load_dataset
from pyserini.setup import configure_classpath

from convert_jsonl_module import encode


SPLITS = ["train", "validation", "test", "other"]
SETTINGS = ["small", "base", "large"]
ENCODING_FIELDS = ["text", "image_caption"]


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyserini_path", required=True, type=str, help="Path to the Pyserini installation")
    parser.add_argument("--anserini_fatjar_path", required=True, type=str, help="Path to the Anserini fatjar")
    parser.add_argument("--images", default="TREC-AToMiC/AToMiC-Images-v0.2", type=str)
    parser.add_argument("--texts",  default="TREC-AToMiC/AToMiC-Texts-v0.2", type=str)
    parser.add_argument("--qrels",  default="TREC-AToMiC/AToMiC-Qrels-v0.2", type=str)
    return parser


def prep_qrels(qrels_ds, split, pyserini_path):
    qrels_dir = pyserini_path / "collections" / "atomic"
    qrels_dir.mkdir(exist_ok=True)

    qrel_ds = load_dataset(qrels_ds, split=split)
    qrel_ds.to_csv(
        qrels_dir / f"{split}.qrels.t2i.projected.trec", header=None, sep=" ", index=False
    )
    qrel_ds.to_csv(
        qrels_dir / f"{split}.qrels.i2t.projected.trec",
        columns=["image_id", "Q0", "text_id", "rel"], header=None, sep=" ", index=False
    )


def anserini_index(indexing_args):
    # this import is deliberately placed here (has to be imported AFTER configure_classpath is run)
    from jnius import autoclass
    IndexCollection = autoclass("io.anserini.index.IndexCollection")
    IndexCollection.main(indexing_args)


def create_index(setting, split, pyserini_path):
    atomic_collections_path = pyserini_path / "collections" / "atomic"
    if setting == "small" and not split:
        raise Exception('Please provide the split when setting="small"')
    postfix = "." + split if setting == "small" else ""

    # create a directory that contains the documents we want to index
    # NOTE: should we consider cleaning up this directory once we are done indexing?
    text_dir = atomic_collections_path / f"text-collection.{setting}{postfix}"
    text_dir.mkdir(exist_ok=True)

    image_dir = atomic_collections_path / f"image-collection.{setting}{postfix}"
    image_dir.mkdir(exist_ok=True)

    if setting == "small":
        text_jsonl_paths = list((atomic_collections_path / "text-collection").glob(f"{split}*.jsonl"))
        image_jsonl_paths = list((atomic_collections_path / "image-collection").glob(f"{split}*.jsonl"))
    elif setting == "base":
        text_jsonl_paths = []
        image_jsonl_paths = []
        for split in ["train", "validation", "test"]:
            text_jsonl_paths.extend(list((atomic_collections_path / "text-collection").glob(f"{split}*.jsonl")))
            image_jsonl_paths.extend(list((atomic_collections_path / "image-collection").glob(f"{split}*.jsonl")))
    elif setting == "large":
        text_jsonl_paths = list((atomic_collections_path / "text-collection").glob("*.jsonl"))
        image_jsonl_paths = list((atomic_collections_path / "image-collection").glob("*.jsonl"))

    for p in text_jsonl_paths:
        (text_dir / p.name).symlink_to(p, target_is_directory=False)

    for p in image_jsonl_paths:
        (image_dir / p.name).symlink_to(p, target_is_directory=False)

    print(f"Indexing text: {setting}{postfix}")
    text_indexing_args = [
        "-input", str(text_dir.resolve()),
        "-collection", "JsonCollection",
        "-index", f"indexes/lucene-index.atomic.text.flat.{setting}{postfix}",
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", "8", "-storePositions", "-storeDocvectors", "-storeRaw",
    ]
    anserini_index(text_indexing_args)

    print(f"Indexing image: {setting}{postfix}")
    image_indexing_args = [
        "-input", str(image_dir.resolve()),
        "-collection", "JsonCollection",
        "-index", f"indexes/lucene-index.atomic.image.flat.{setting}{postfix}",
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", "8", "-storePositions", "-storeDocvectors", "-storeRaw",
    ]
    anserini_index(image_indexing_args)


def anserini_search(search_args):
    # this import is deliberately placed here (has to be imported AFTER configure_classpath is run)
    from jnius import autoclass
    SearchCollection = autoclass("io.anserini.search.SearchCollection")
    SearchCollection.main(search_args)


def search_anserini(split, setting, pyserini_path):
    postfix = "." + split if setting == "small" else ""

    indexes = pyserini_path / "indexes"
    text_index_dir = indexes / f"lucene-index.atomic.text.flat.{setting}{postfix}"
    image_index_dir = indexes / f"lucene-index.atomic.image.flat.{setting}{postfix}"

    runs = pyserini_path / "runs"
    runs.mkdir(exist_ok=True)
    t2i_run_dir = runs / f"run.{split}.bm25-anserini-default.t2i.{setting}.trec"
    i2t_run_dir = runs / f"run.{split}.bm25-anserini-default.i2t.{setting}.trec"

    i2t_search_args = [
        "-index", str(text_index_dir.resolve()),
        "-topics", str((pyserini_path / "collections" / "atomic" / f"image-collection.{setting}{postfix}/{split}.image-caption.jsonl").resolve()),
        "-topicreader", "JsonString",
        "-topicfield", "contents",
        "-output", str(i2t_run_dir.resolve()),
        "-bm25", "-hits", "1000", "-parallelism", "64", "-threads", "64"
    ]
    anserini_search(i2t_search_args)

    t2i_search_args = [
        "-index", str(image_index_dir.resolve()),
        "-topics", str((pyserini_path / "collections" / "atomic" / f"text-collection.{setting}{postfix}/{split}.text.jsonl").resolve()),
        "-topicreader", "JsonString",
        "-topicfield", "contents",
        "-output", str(t2i_run_dir.resolve()),
        "-bm25", "-hits", "1000", "-parallelism", "64", "-threads", "64"
    ]
    anserini_search(t2i_search_args)

    """
    TODO: seprate out this logic for evaluating in a different script
    echo "==== Text2Image ===="
    t2i_qrel="${QREL_DIR}/${SPLIT}.qrels.t2i.projected.trec"
    $PATH_TREC_EVAL -c -m recip_rank -M 10 ${t2i_qrel} ${t2i_run}
    $PATH_TREC_EVAL -c -m recall.10,1000 ${t2i_qrel} ${t2i_run} 

    echo "==== Image2Text ===="
    i2t_qrel="${QREL_DIR}/${SPLIT}.qrels.i2t.projected.trec"
    $PATH_TREC_EVAL -c -m recip_rank -M 10 ${i2t_qrel} ${i2t_run}
    $PATH_TREC_EVAL -c -m recall.10,1000 ${i2t_qrel} ${i2t_run}
    """


def main(args):
    configure_classpath(args.anserini_fatjar_path)
    pyserini_path = Path(args.pyserini_path)

    for split in SPLITS:
        if split == "other":
            continue
        print(f"RUN PREP QRELS, split: {split}")
        prep_qrels(args.qrels, split, pyserini_path)

    for split in SPLITS:
        for encoding_field in ENCODING_FIELDS:
            print(f"RUN ENCODE, split: {split}, encoding_field: {encoding_field}")
            encode(split, encoding_field, args.qrels, args.images, args.texts, pyserini_path / "collections" / "atomic")

    for setting in SETTINGS:
        print(f"RUN CREATE INDEX, setting: {setting}")
        if setting == "small":
            create_index(setting, "validation", pyserini_path)
            create_index(setting, "test", pyserini_path)
        else:
            create_index(setting, None, pyserini_path)

    for setting in SETTINGS:
        print(f"RUN SEARCH, setting: {setting}")
        search_anserini("validation", setting, pyserini_path)


if __name__ == "__main__":
    # TODO: maybe can add a command line argument to skip qrels, encoding, indexing
    # Sample run command:
    # python run_bm25_baseline.py --pyserini_path "/home/kj25lee/scratch/pyserini" --anserini_fatjar_path "/home/kj25lee/scratch/pyserini/resources/jars"
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
