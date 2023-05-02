import os
import sys
import math
import numpy as np
import argparse

import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

from io import BytesIO
from pathlib import Path
from encoders import ClipEncoder
from tqdm.auto import tqdm

import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

module_path = str(Path(__file__).absolute().parents[2])
sys.path.append(module_path)
os.environ['NUMEXPR_MAX_THREADS'] = '8'

from src.data import AtomicDataset


def get_args_parser():
    parser = argparse.ArgumentParser('Encode embeddings', add_help=False)
    parser.add_argument('--inputs', type=str, default='TREC-AToMiC/AToMiC-Texts-v0.2.1')
    parser.add_argument('--encode_type', type=str, default='text', choices=['text', 'image'])
    parser.add_argument('--id_column', type=str, default='text_id')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--qrels', type=str, default='TREC-AToMiC/AToMiC-Qrels-v0.2')
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--shard_num', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='openai/clip-vit-base-patch32', help="encoder name")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--output_dir', type=str, help='directory to store embeddings')

    return parser


class TextCollator:
    def __init__(self, id_col, field_col, processor, prompt=None):
        self.id_col = id_col
        self.field_col = field_col
        self.processor = processor
        self.prompt = prompt
    
    def __call__(self, batch):
        batch_ids, batch_inputs = [], []
        for item in batch:
            batch_ids.append(item[self.id_col])
            content = []
            if isinstance(self.field_col, list):
                for field in self.field_col:
                    # if the column val is a list, concat again
                    if isinstance(item[field], list):
                        content.append(' '.join(item[field]))
                    else:
                        content.append(item[field])
            else:
                content.append(item[self.field_col])
            
            text = ' '.join(content)
            batch_inputs.append(text)
        
        if self.prompt:
            _inputs = []
            for item in batch_inputs:
                _inputs.append(f'{self.prompt}{item}')
            batch_inputs = _inputs
        else:
            batch_inputs = batch_inputs

        batch_inputs = self.processor(
            text=batch_inputs,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=77,
            truncation=True,
            return_token_type_ids=False,
        )

        return {
            'id': batch_ids,
            'text': batch_inputs,
        }

class Transform(nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = nn.Sequential(
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


class ImageCollator:
    def __init__(self, id_col, field_col, processor):
        self.id_col = id_col
        self.field_col = field_col

        transforms = Transform(
            [processor.crop_size['height'], processor.crop_size['width']],
            processor.image_mean,
            processor.image_std,
        )
        self.transforms = torch.jit.script(transforms)
    
    def __call__(self, batch):
        batch_ids, batch_inputs = [], []
        
        for item in batch:
            batch_ids.append(item[self.id_col])
            batch_inputs.append(self.transforms(ToTensor()(item[self.field_col])))
        
        return {
            'id': batch_ids,
            'image': {'pixel_values': torch.stack(batch_inputs)},
        }


class NumpyWriter:
    def __init__(self, dir_path, filename, shard_id, shard_num):
        self.dir_path = dir_path
        self.id_filename = 'ids'
        self.filename = filename
        self.shard_id = shard_id
        self.shard_num = shard_num
        self.shard_digit = int(math.log10(shard_num)) + 1
        self.id_file = None
        self.__init_batch()

    def __enter__(self):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        shard_str = f"{str(self.shard_id).zfill(self.shard_digit)}-of-{str(self.shard_num).zfill(self.shard_digit)}"
        self.id_file = Path(self.dir_path, f"{self.id_filename}.{shard_str}.txt").open('w')
        self.file    = Path(self.dir_path, f"{self.filename}.{shard_str}.npy").open("wb")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.id_file.close()
        self.file.close()
    
    def __init_batch(self):
        self.ids = []
        self.embeddings = []
    
    def add(self, batch):
        self.ids.extend(batch['id'])
        self.embeddings.append(batch['vector'])
    
    def write(self):
        logging.info('Writing files ...')
        for _id in self.ids:
            self.id_file.write(f'{_id}\n')
        
        embedding_matrix = np.concatenate(self.embeddings)
        npb = BytesIO()
        np.save(npb, embedding_matrix)
        self.file.write(npb.getbuffer())
        logging.info('Done')


def main(args):

    logging.info(f"{args}")

    logging.info('Prepare model')
    encoder = ClipEncoder(
        args.encoder,
        encode_type=args.encode_type, 
        device=args.device,
        dtype=args.dtype,
        l2_norm=True,
    )

    logging.info('Prepare data')
    dataset = AtomicDataset(
        data_name_or_path=args.inputs,
        id_column=args.id_column,
        qrel_name_or_path=args.qrels,
    )

    if args.split:
        dataset = dataset.get_split(args.split)
    
    # shard the data
    dataset = dataset.shard(index=args.shard_id, num_shards=args.shard_num)

    if args.encode_type == 'text':
        collator = TextCollator(
            id_col='text_id',
            field_col=['page_title', 'section_title', 'hierachy', 'context_section_description', 'context_page_description'],
            processor=encoder.processor.tokenizer,
        )
    else:
        collator = ImageCollator(
            id_col='image_id',
            field_col='image',
            processor=encoder.processor.image_processor,
        )
    
    iterator = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logging.info('Encode data')
    output_dir = Path(args.output_dir, args.encode_type, args.split)    
    filename = 'embeddings'
    writer = NumpyWriter(dir_path=output_dir, filename=filename, shard_id=args.shard_id, shard_num=args.shard_num)
    
    with writer:
        for batch in tqdm(iterator, total=len(iterator), desc='encode ...'):
            embeddings = encoder.encode(batch)
            batch['vector'] = embeddings
            writer.add(batch)
        writer.write()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)