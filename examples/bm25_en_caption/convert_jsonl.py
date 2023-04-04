import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk, concatenate_datasets
from textacy import preprocessing




def get_args_parser():
    parser = argparse.ArgumentParser('image-parquets', add_help=False)
    parser.add_argument('--images', default='TREC-AToMiC/AToMiC-Images-v0.2', type=str)
    parser.add_argument('--texts',  default='TREC-AToMiC/AToMiC-Texts-v0.2', type=str)
    parser.add_argument('--qrels',  default='TREC-AToMiC/AToMiC-Qrels-v0.2', type=str)
    parser.add_argument('--encode_field', choices=['image_caption', 'text'])
    parser.add_argument('--output_dir', default='collection', type=str)
    parser.add_argument('--split', choices=['train', 'validation', 'test', 'other'])
    return parser


def get_pipeline():
    preproc = preprocessing.make_pipeline(
                preprocessing.remove.html_tags,
                preprocessing.normalize.unicode,
                preprocessing.normalize.whitespace,
                preprocessing.normalize.bullet_points,
                preprocessing.normalize.quotation_marks,
                preprocessing.replace.hashtags,
                preprocessing.remove.brackets,
                preprocessing.replace.urls
              )
    return preproc


def convert_flatten_image(example, id_col):
    result = dict()
    result['id'] = example[id_col]
    pipe = get_pipeline()
    temp = []
    # filter en
    en_pos = []
    for pos, lang_id in enumerate(example['language']):
        if lang_id == 'en':
            en_pos.append(True)
        else:
            en_pos.append(False)

    for key, val in example.items():
        if key == id_col:
            continue
        if key == 'language':
            continue

        content = []
        if isinstance(val, list):
            for item, valid in zip(val, en_pos):
                if valid:
                    clean = pipe(item)
                    content.append(clean)
        else:
            clean = pipe(val)
            content.append(clean)
        text = ' '.join(content)
        result[key] = text
        temp.append(text)
    result['contents'] = ' '.join(temp)
    return result


def convert_flatten_text(example, id_col):
    result = dict()
    result['id'] = example[id_col]
    result['contents'] = ''
    pipe = get_pipeline()
    temp = []
    for key, val in example.items():
        if key == id_col:
            continue
        
        content = []
        if isinstance(val, list):
            for item in val:
                clean = pipe(item)
                content.append(clean)
        else:
            clean = pipe(val)
            content.append(clean)
        text = ' '.join(content)
        result[key] = text
        temp.append(text)
    result['contents'] = ' '.join(temp)
    return result

def limit_clause(example, limit=1024):
    # truncate the number of 'words' by space
    clauses = example['contents'].split()
    return {'contents': ' '.join(clauses[:limit])}


def maybe_shard_to_save(dataset, output_path, num_lines=1000000):
    file_path = Path(output_path)
    if len(dataset) > num_lines:
        n_shards = len(dataset) // num_lines + 1
        for idx in range(n_shards):
            subset = dataset.shard(num_shards=n_shards, index=idx)
            output_file = Path(file_path.parent, file_path.stem + f'.part-{idx:02d}.jsonl')
            subset.to_json(str(output_file))
    else:
        dataset.to_json(str(file_path))


def main(args):
    
    if args.split == 'other':
        qrels = load_dataset(args.qrels)
        qrels = concatenate_datasets([qrels[split] for split in ['train', 'validation', 'test']])
    else:
        qrels = load_dataset(args.qrels, split=args.split)

    if args.encode_field == 'image_caption':

        valid_images = set(qrels.unique('image_id'))

        images = load_dataset(args.images, split='train')
        images = images.remove_columns(['image', 'image_url']) # remove images for faster processing
        old_col = images.column_names

        if args.split == 'other':
            images = images.filter(lambda example: bool(example['image_id'] not in valid_images), num_proc=32)
        else:
            images = images.filter(lambda example: bool(example['image_id'] in valid_images), num_proc=32)

        images = images.map(convert_flatten_image, fn_kwargs={"id_col": 'image_id'}, num_proc=16)
        images = images.map(limit_clause, num_proc=16)
        images = images.remove_columns(old_col)
        maybe_shard_to_save(images, f'image-{args.output_dir}/{args.split}.image-caption.jsonl')
    
    if args.encode_field == 'text':
        valid_texts = set(qrels.unique('text_id'))

        texts = load_dataset(args.texts, split='train')
        texts = texts.remove_columns(['media', 'category', 'source_id', 'page_url'])
        old_col = texts.column_names
        
        if args.split == 'other':
            texts = texts.filter(lambda example: bool(example['text_id'] not in valid_texts), num_proc=32)
        else:
            texts = texts.filter(lambda example: bool(example['text_id'] in valid_texts), num_proc=32)
        
        texts = texts.map(convert_flatten_text, fn_kwargs={"id_col": "text_id"}, num_proc=16)
        texts = texts.map(limit_clause, num_proc=16)
        texts = texts.remove_columns(old_col)
        maybe_shard_to_save(texts, f'text-{args.output_dir}/{args.split}.text.jsonl')
    
    
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    main(args)
