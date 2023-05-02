import argparse
import faiss
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_args_parser():
    parser = argparse.ArgumentParser('Index embeddings', add_help=False)
    parser.add_argument('--embedding_dir', type=str)
    parser.add_argument('--index', type=str)
    parser.add_argument('--index_type', type=str, default='flat')
    return parser


class NumpyReader:
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir
        vec_files = list(Path(self.embedding_dir).glob('embeddings*.npy'))
        self.vec_files = sorted(vec_files)
        id_files = Path(embedding_dir).glob('ids*.txt')
        self.id_files = sorted(id_files)
        self.load_embeddings()
        self.dim = self.vectors.shape[1]

        
    def load_embeddings(self):
        self.vectors = []
        self.ids = []
        for f in tqdm(self.vec_files, total=len(self.vec_files)):
            self.vectors.append(np.load(f))
        
        self.vectors = np.concatenate(self.vectors)
    
        for _id in tqdm(self.id_files, total=len(self.id_files)):
            with open(_id, 'r') as f_id:
                _ids = [l.strip('\n') for l in f_id.readlines()]
                self.ids.extend(_ids)   
    
    def __iter__(self):
        for _id, vec in zip(self.ids, self.vectors):
            yield {'id': _id, 'vector': vec}
    
    def __len__(self):
        return len(self.ids)



def main(args):

    logging.info('Prepare faiss index')
    embedding_reader = NumpyReader(args.embedding_dir)
    vector_dim = embedding_reader.dim

    if args.index_type == 'flat':
        index_factory = faiss.IndexFlatIP(vector_dim)
    
    index_factory.verbose = True

    logging.info('Add vectors to faiss index')
    _ids = []
    for item in tqdm(embedding_reader):
        vector = item['vector']
        index_factory.add(vector.reshape((1, -1)).astype('float32'))
        _ids.append(item['id'])

    output_dir = Path(f"{args.index}.faiss.{args.index_type}")
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    faiss.write_index(index_factory, str(Path(output_dir, 'index')))
    
    with Path(output_dir, 'docid').open('w') as f_id:
        for _id in _ids:
            f_id.write(f"{_id}\n")

    logging.info('Done')



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)