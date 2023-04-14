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
import os

from datasets import load_dataset


SPLITS = ["train", "validation", "test", "other"]
SETTINGS = ["small", "base", "large"]


def prep_qrels(split):
    qrel_ds = load_dataset('TREC-AToMiC/AToMiC-Qrels-v0.2', split=split)
    qrel_ds.to_csv(
        f"{split}.qrels.t2i.projected.trec", header=None, sep=" ", index=False
    )
    qrel_ds.to_csv(
        f"{split}.qrels.t2i.projected.trec",
        columns=["image_id", "Q0", "text_id", "rel"], header=None, sep=" ", index=False
    )


def prep_encodings(split, encode_field):
    cmd = f"python convert_jsonl.py --split {split} --encode_field {encode_field}"
    os.system(cmd)


def create_index(setting):
    cmd = f"./index_anserini.sh index {setting}"
    os.system(cmd)


def search_anserini(split, setting):
    cmd = f"./search_anserini.sh {split} {setting}"
    os.system(cmd)


def main():
    # TODO: maybe can add a command line argument to skip qrels, encoding, and indexing
    #       (or altogether separate them out in a separate command)
    for split in SPLITS:
        prep_qrels(split)

    for split in SPLITS:
        for encoding_field in ["text", "image_caption"]:
            prep_encodings(split, encoding_field)

    for setting in SETTINGS:
        create_index(setting)

    for setting in SETTINGS:
        print(f"Evaluate on setting {setting} (validation)")
        search_anserini("validation", setting)

    # NOTE: should we do cleanup? e.g. remove any files/ symlinks we created while evaluating
    # NOTE: also need to take care of setting the environment variables in the shell scripts


if __name__ == "__main__":
    main()
