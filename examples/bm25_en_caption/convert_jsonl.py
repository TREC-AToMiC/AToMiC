from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from textacy import preprocessing


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
    for _, lang_id in enumerate(example['language']):
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(dataset) > num_lines:
        n_shards = len(dataset) // num_lines + 1
        for idx in range(n_shards):
            subset = dataset.shard(num_shards=n_shards, index=idx)
            output_file = Path(output_path.parent, output_path.stem + f'.part-{idx:02d}.jsonl')
            subset.to_json(str(output_file))
    else:
        dataset.to_json(str(output_path))


def encode(split, encode_field, qrels_ds, image_ds, text_ds, output_path):
    if split == 'other':
        qrels = load_dataset(qrels_ds)
        qrels = concatenate_datasets(
            [qrels[qrels_split] for qrels_split in ['train', 'validation', 'test']]
        )
    else:
        qrels = load_dataset(qrels_ds, split=split)

    if encode_field == 'image_caption':
        valid_images = set(qrels.unique('image_id'))

        images = load_dataset(image_ds, split='train')
        # remove images for faster processing
        images = images.remove_columns(['image', 'image_url'])
        old_col = images.column_names

        if split == 'other':
            images = images.filter(lambda example: example['image_id'] not in valid_images, num_proc=32)
        else:
            images = images.filter(lambda example: example['image_id'] in valid_images, num_proc=32)

        images = images.map(convert_flatten_image, fn_kwargs={'id_col': 'image_id'}, num_proc=16)
        images = images.map(limit_clause, num_proc=16)
        images = images.remove_columns(old_col)
        maybe_shard_to_save(images, output_path / 'image-collection' / f'{split}.image-caption.jsonl')

    elif encode_field == 'text':
        valid_texts = set(qrels.unique('text_id'))

        texts = load_dataset(text_ds, split='train')
        texts = texts.remove_columns(['media', 'category', 'source_id', 'page_url'])
        old_col = texts.column_names

        if split == 'other':
            texts = texts.filter(lambda example: bool(example['text_id'] not in valid_texts), num_proc=32)
        else:
            texts = texts.filter(lambda example: bool(example['text_id'] in valid_texts), num_proc=32)

        texts = texts.map(convert_flatten_text, fn_kwargs={'id_col': 'text_id'}, num_proc=16)
        texts = texts.map(limit_clause, num_proc=16)
        texts = texts.remove_columns(old_col)
        maybe_shard_to_save(texts, output_path / 'text-collection' / f'{split}.text.jsonl')
