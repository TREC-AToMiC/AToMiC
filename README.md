# AToMiC
The AToMiC dataset is a large-scale image/text retrieval test collection designed to aid in multimedia content creation.
The dataset is composed of approximately 10 million images and texts, and there are 4 million image--text binary relevance judgments available.
We formulate two retrieval tasks: image suggestion and image promotion. 
The objectives of these tasks are to identify images that complement the textual content and identify text that corresponds to the image.

Learn more about AToMiC Dataset from our [arxiv paper](https://arxiv.org/abs/2304.01961).

![AToMiC](asset/teaser.png)

## Usage
```
pip install datasets
```

```python
from datasets import load_dataset

dataset = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train')
print(dataset[0])
```

## Latest Update
2023 May:
- [Call for participants](https://trec.nist.gov/pubs/call2023.html): We are excited to announce that we will be hosting a dedicated track at the TREC 2023 workshop. We invite all interested participants to submit their runs and join us.
- Text collection update: We have addressed the missing entity issues in our text collection and have released an updated version, `AToMiC-Texts-v0.2.1`. For those interested in participating in the TREC2023 evaluation, please use this version. If you wish to reproduce the results presented in our SIGIR paper, please use `AToMiC-Texts-v0.2`. We have created a spreadsheet highlighting the differences in retrieval effectiveness between the two versions, which can be found [here](https://docs.google.com/spreadsheets/d/1wSi_79Qx3GA1WAirwvoapiWJ4m2bPRM_rtUWRZ2qRIo/edit?usp=sharing).

## Getting Started
We can use HuggingFace's _Datasets_ and _Transformers_ to explore the AToMiC Dataset.
You can find their great documentation in the following links: 
- [Transformers](https://huggingface.co/transformers/index.html): >=4.26.0
- [Datasets](https://huggingface.co/docs/datasets/index.html): >=2.8.0

To get started with AToMiC Dataset, we refer you to the following locations:
- [Notebooks](https://github.com/TREC-AToMiC/AToMiC/tree/main/notebooks): a series notebooks for playing with AToMiC with 🤗 _Datasets_ and _Transformers_

## Text collection
[🤗 Datasets](https://huggingface.co/datasets/TREC-AToMiC/AToMiC-Texts-v0.2.1)

The files are stored in Parquet format. Each row in the file corresponds to a Wikipedia section prepared from the 20221101 English Wikipedia XML dump.
The basic data fields are: `page_title`, `hierachy`, `section_title`, `context_page_description`, and `context_section_description`.
There are other fields such as `media`, `category`, and `source_id` for our internal usage.
Note that we set `Introduction` for as the section title for leading sections.
The total size of the text collection is approximately 14 GB.

## Image collection
[🤗 Datasets](https://huggingface.co/datasets/TREC-AToMiC/AToMiC-Images-v0.2)

The images are stored in the Parquet format, with each row representing an image that has been crawled from the Wikimedia Commons database.
The image data is stored as bytes of a `PIL.WebPImagePlugin.WebPImageFile` object, along with other metadata including `reference`, `alt-text`, and `attribution` captions. 
Additionally, the `language_id` field provides a list of language identifiers indicating the language of the Wikipedia captions for each image.
Please note that the total size of the image collection is approximately 180 GB.

## Sparse relevance judgements
[🤗 Datasets](https://huggingface.co/datasets/TREC-AToMiC/AToMiC-Qrels-v0.2)

The relevance judgments are formatted in standard TREC qrels format, as follows:
```
text_id Q0 image_id relevance
```
The default setting of the Qrels is for text-to-image retrieval task.
Each row in the Qrel file stands for the relavant image--text pairs in the text and image collections.
To faciliate the image-to-text retrieval task, you only need to swap the position of `text_id` and `image_id`.

## Citations
If you find this resource useful, please consider citing our paper and the WIT paper.

```bibtex
@article{yang2023atomic,
      title={AToMiC: An Image/Text Retrieval Test Collection to Support Multimedia Content Creation}, 
      author={Jheng-Hong Yang and Carlos Lassance and Rafael Sampaio de Rezende and Krishna Srinivasan and Miriam Redi and Stéphane Clinchant and Jimmy Lin},
      journal={arXiv preprint 2304.01961},
      year={2023},
}
```

```bibtex
@article{srinivasan2021wit,
  title={WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning},
  author={Srinivasan, Krishna and Raman, Karthik and Chen, Jiecao and Bendersky, Michael and Najork, Marc},
  journal={arXiv preprint arXiv:2103.01913},
  year={2021}
}
```

## License

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## Contact
If any questions, please contact:
jheng-hong.yang@uwaterloo.ca or trec-atomic-organizers@googlegroups.com
