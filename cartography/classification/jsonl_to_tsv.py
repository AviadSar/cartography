import pandas as pd
import os


def jsonl_to_tsv(jsonl_path, tsv_path):
    df = pd.read_json(jsonl_path, lines=True)
    df.to_csv(tsv_path, sep='\t')


# tsv_dir = r'C:\my_documents\datasets\WINOGRANDE_TSV'
tsv_dir = r'/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV'
if not os.path.exists(tsv_dir):
    os.makedirs(tsv_dir)


# jsonl_to_tsv(r'C:\my_documents\datasets\winogrande_1.1\winogrande_1.1\test.jsonl', r'C:\my_documents\datasets\WINOGRANDE_TSV\test.tsv')

jsonl_to_tsv(r'/cs/labs/roys/aviadsa/datasets/winogrande_1.1/winogrande_1.1/dev.jsonl', r'/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/dev.tsv')
jsonl_to_tsv(r'/cs/labs/roys/aviadsa/datasets/winogrande_1.1/winogrande_1.1/test.jsonl', r'/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/test.tsv')
jsonl_to_tsv(r'/cs/labs/roys/aviadsa/datasets/winogrande_1.1/winogrande_1.1/train_xl.jsonl', r'/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/train.tsv')