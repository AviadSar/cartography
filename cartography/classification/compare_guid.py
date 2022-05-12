import argparse
import os
from cartography.data_utils_glue import read_glue_tsv

parser = argparse.ArgumentParser()

parser.add_argument("--data_file_1",
                    default="",
                    type=os.path.abspath,
                    help="1st data file to compare")
parser.add_argument("--data_file_2",
                    default="",
                    type=os.path.abspath,
                    help="2nd data file to compare")

args = parser.parse_args()

# lines = 0
# with open(args.data_file_1, 'r') as tsv_file:
#     for line in tsv_file:
#         lines += 1

args.data_file_1 = '../filtered_datasets/cartography_variability_0.33/SNLI/huji_snli_5_roberta/train.tsv'
args.data_file_2 = '../filtered_datasets/cartography_variability_0.33/SNLI/huji_snli_5_bert/train.tsv'

data1 = read_glue_tsv(args.data_file_1, guid_index=8)
data2 = read_glue_tsv(args.data_file_2, guid_index=8)

print('Number of examples 1: ', len(data1[0].keys()))
print('Number of examples 2: ', len(data2[0].keys()))
print('Intersection size: ', len(set(data1[0].keys()).intersection(set(data2[0].keys()))))
a = 1
