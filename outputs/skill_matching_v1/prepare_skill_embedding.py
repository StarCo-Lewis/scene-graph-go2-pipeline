"""Explicit, one-time download into an isolated embedding environment."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('Output already exists; use the existing model or a new directory')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device='cpu')
    model.save(str(args.output_dir))
    print(f'Local model saved: {args.output_dir}')


if __name__ == '__main__':
    main()
