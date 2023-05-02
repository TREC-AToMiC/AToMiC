import argparse
from pathlib import Path
from datasets import load_dataset




def get_args_parser():
    parser = argparse.ArgumentParser('Index embeddings', add_help=False)
    parser.add_argument('--qrels', type=str, default='TREC-AToMiC/AToMiC-Qrels-v0.2')
    parser.add_argument('--output_dir', type=str, default=Path(__file__))
    return parser

def main(args):
    qrels_dir = Path(args.output_dir, "qrels")
    qrels_dir.mkdir(exist_ok=True)

    qrels = load_dataset(args.qrels)
    for split in ['train', 'validation', 'test']:
        qrels[split].to_csv(
            Path(qrels_dir, f"{split}.qrels.t2i.projected.trec"),
            header=None, sep=" ", index=False
        )
        
        qrels[split].to_csv(
            Path(qrels_dir, f"{split}.qrels.i2t.projected.trec"),
            columns=["image_id", "Q0", "text_id", "rel"],
            header=None, sep=" ", index=False
        )

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)