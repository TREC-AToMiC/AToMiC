import argparse
import numpy as np
from pathlib import Path
from pyserini.search.faiss import FaissSearcher
from pyserini.output_writer import OutputFormat, get_output_writer, tie_breaker
from tqdm.auto import tqdm

from index import NumpyReader



def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)
    # Model settings
    parser.add_argument('--index', type=str)
    parser.add_argument('--topics', type=str)
    parser.add_argument('--encoder', type=str, default='openai/clip-vit-base-patch32', help="encoder name")
    parser.add_argument('--hits', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--output', type=str, metavar='path')
    parser.add_argument('--output-format', type=str, metavar='format', default=OutputFormat.TREC.value)
    parser.add_argument('--output_tag', type=str, default='atomic')
    return parser



def main(args):
    searcher = FaissSearcher(
        index_dir=args.index,
        query_encoder=args.encoder,
    )
    topics = NumpyReader(args.topics)

    output_path = args.output
    if output_path is None:
        tokens = ['run', args.output_tag, 'clip', 'txt']
        output_path = '.'.join(tokens)

    print(f'Running {args.topics} topics, saving to {output_path}...')
    tag = output_path[:-4] if args.output is None else 'Pyserini'

    output_writer = get_output_writer(
        output_path, OutputFormat(args.output_format), 'w',
        max_hits=args.hits, tag=tag
    )

    with output_writer:
        batch_topics = list()
        batch_topic_ids = list()
        for index, item in enumerate(tqdm(topics, total=len(topics))):
            topic_id = item['id']
            topic_vec = item['vector']
            
            batch_topic_ids.append(str(topic_id))
            batch_topics.append(topic_vec.reshape(1, -1))
            if (index + 1) % args.batch_size == 0 or index == len(topic_vec) - 1:
                
                results = searcher.batch_search(
                    queries=np.concatenate(batch_topics),
                    q_ids=batch_topic_ids,
                    k=args.hits,
                    threads=args.threads,
                    return_vector=False
                )
                
                results = [(id_, results[id_]) for id_ in batch_topic_ids]
                batch_topic_ids.clear()
                batch_topics.clear()
            else:
                continue
        
            for topic, hits in results:
                output_writer.write(topic, tie_breaker(hits))
            
            results.clear()



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)