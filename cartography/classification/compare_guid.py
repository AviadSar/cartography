import sys
if sys.path[0] != '.':
    print('first path variable is: ' + sys.path[0])
    sys.path.insert(0, '.')
    print("added '.' to sys.path")

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

args.data_file_0 = '/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.33/WINOGRANDE/huji_winogrande_roberta_large_5_epochs/train.tsv'
args.data_file_1 = '/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.33/WINOGRANDE/huji_winogrande_deberta_large_5_epochs/train.tsv'
args.data_file_2 = '/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.33/WINOGRANDE/huji_winogrande_electra_5_epochs/train.tsv'

data0 = read_glue_tsv(args.data_file_0, guid_index=0)
data1 = read_glue_tsv(args.data_file_1, guid_index=0)
data2 = read_glue_tsv(args.data_file_2, guid_index=0)

datas = [data0, data1, data2]

for i, data in enumerate(datas):
    print('Number of examples in data {}: '.format(i), len(data[0].keys()))

print('Intersection size: ', len(set(datas[0][0].keys()).intersection(set(datas[1][0].keys()))))
print('Intersection size: ', len(set(datas[0][0].keys()).
                                 intersection(set(datas[1][0].keys())).
                                 intersection(set(datas[2][0].keys()))))
a = 1
