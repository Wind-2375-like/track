"""
Upload TRACK benchmark datasets to HuggingFace Hub.

Converts the three test_500.pkl files (wiki/grow, code, math) into a HuggingFace
DatasetDict with three configs. Each example has five fields aligned with the paper:
  - question       : complex reasoning question (q)
  - answer         : final answer (a)
  - probing_questions : list of probing questions (q_i)
  - probing_answers   : list of probing answers (a_i)
  - atomic_facts      : list of atomic facts (K_q)
"""

import os
import sys
import json
import pickle
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from datasets import Dataset, DatasetDict


def load_and_convert(pkl_path: str) -> Dataset:
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    records = []
    for item in raw:
        probing_questions = []
        probing_answers = []
        atomic_facts = []
        for pq in item.get('probe_questions', []):
            probing_questions.append(pq.get('question', ''))
            probing_answers.append(pq.get('answer', ''))
            atomic_facts.append(pq.get('knowledge', ''))

        records.append({
            'question': str(item.get('multihop_question', '')),
            'answer': str(item.get('multihop_answer', '')),
            'probing_questions': probing_questions,
            'probing_answers': probing_answers,
            'atomic_facts': atomic_facts,
        })

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="Upload TRACK benchmark to HuggingFace Hub")
    parser.add_argument('--data_dir', type=str, default='data',
                        help="Root data directory (default: data)")
    parser.add_argument('--repo_id', type=str, default='yiyangfengSBU/track',
                        help="HuggingFace repo id (default: yiyangfengSBU/track)")
    parser.add_argument('--api_config_file', type=str, default='./api_key/config.json',
                        help="Path to api_key/config.json")
    args = parser.parse_args()

    # Load HF token
    with open(args.api_config_file, 'r') as f:
        config = json.load(f)
    hf_token = config.get('api_key', {}).get('huggingface_api_key')
    if not hf_token:
        raise ValueError("huggingface_api_key not found in api_key/config.json")

    # grow folder → wiki config
    splits = {
        'wiki': os.path.join(args.data_dir, 'grow', 'test_500.pkl'),
        'code': os.path.join(args.data_dir, 'code', 'test_500.pkl'),
        'math': os.path.join(args.data_dir, 'math', 'test_500.pkl'),
    }

    for name, path in splits.items():
        print(f"Loading {name} from {path} ...")
        ds = load_and_convert(path)
        print(f"  {len(ds)} examples, columns: {ds.column_names}")
        print(f"  Uploading {name} config to {args.repo_id} ...")
        ds.push_to_hub(
            repo_id=args.repo_id,
            config_name=name,
            split='test',
            token=hf_token,
        )
        print(f"  Done: {name}")

    print(f"\nAll three splits uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == '__main__':
    main()
