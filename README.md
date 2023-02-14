# AToMiC
A Text/Image Retrieval Test Collection to Support Multimedia Content Creation

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## Usage
```
pip install datasets
```

```python
from datasets import load_dataset

dataset = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train')
print(dataset[0])
```

## Text collection
[🤗 Datasets](https://huggingface.co/datasets/TREC-AToMiC/AToMiC-Texts-v0.2)

The files are in parquet format.
Each row in the file corresponds to a Wikipedia section.
This collection is prepared from the 20221101 English Wikipedia XML dump.
The basic data fields are: `page_title`, `hierachy`, `section_title`, `context_page_description`, and `context_section_description`.
There are other fields such as `media`, `category`, and `source_id` for our internal usage.
Note that we set `Introduction` for as the section title for leading sections.

## Image collection
[🤗 Datasets](https://huggingface.co/datasets/TREC-AToMiC/AToMiC-Images-v0.2)

The files are in parquet format.
Each row in the file corresponds to a image crawled from the Wikicommon database.
The images are stored as the bytes of the `PIL.WebPImagePlugin.WebPImageFile` object.
Other than the images, we also provide other metadata such as reference, alt-text, and attribution captions.
The langauge field contains the list of `language_id' corresponding to which language of Wikipedia is the captions coming from.

## Sparse relevance judgements
[🤗 Datasets](https://huggingface.co/datasets/TREC-AToMiC/AToMiC-Qrels-v0.2)

The relevance judgments are formatted in standard TREC qrels format, as follows:
```
text_id Q0 image_id relevance
```
The default setting of the Qrels is for text-to-image retrieval task.
To faciliate the image-to-text retrieval task, you only need to swap the position of `text_id' and `image_id'.
