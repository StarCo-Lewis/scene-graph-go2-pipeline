"""Isolated CPU embedding service. Model must have been downloaded explicitly."""
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    args = p.parse_args()
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    import torch
    model = SentenceTransformer(args.model_path, device='cpu', local_files_only=True)
    print(json.dumps({'ready': True, 'max_seq_length': model.max_seq_length,
        'sentence_transformers_version': sentence_transformers.__version__,
        'torch_version': torch.__version__, 'python_version': sys.version}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        texts = request['texts']
        # Reject silent truncation: candidate semantics may be in a suffix.
        tokens = model.tokenizer(texts, truncation=False)['input_ids']
        if any(len(t) > model.max_seq_length for t in tokens):
            print(json.dumps({'error': 'EMBEDDING_INPUT_TOO_LONG'}), flush=True)
            continue
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        print(json.dumps({'vectors': vectors}), flush=True)


if __name__ == '__main__':
    main()
