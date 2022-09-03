import sys
if sys.path[0] != '.':
    print('first path variable is: ' + sys.path[0])
    sys.path.insert(0, '.')
    print("added '.' to sys.path")

import argparse
import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import functools
from cartography.data_utils_glue import read_glue_tsv


def compare_guid():
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


def variability_statistics():
    data_file = r'C:\Users\aavia\PycharmProjects\cartography\tmp_files\td_metrics.jsonl'
    with open(data_file, 'r') as data_file:
        lines = [json.loads(line) for line in data_file]

    variabilities = [line['confidence'] for line in lines]
    variabilities.sort()
    n, bins, patches = plt.hist(variabilities, bins=100)
    plt.show()


def compare_correct_samples():
    data_file1 = r'/cs/labs/roys/aviadsa/cartography/outputs/WINOGRANDE/deberta-large/huji_WINOGRANDE_deberta-large_128_batch_12_evals/predictions_winogrande_dev_in_training.lst'
    # data_file2 = r'/cs/labs/roys/aviadsa/cartography/outputs/WINOGRANDE/deberta-large/huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.33_bias_2_12_evals/predictions_winogrande_dev_in_training.lst'
    # data_file2 = r'/cs/labs/roys/aviadsa/cartography/outputs/WINOGRANDE/deberta-large/huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.50_12_evals/predictions_winogrande_dev_in_training.lst'
    data_file2 = r'/cs/labs/roys/aviadsa/cartography/outputs/WINOGRANDE/deberta-large/huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.50_bias_3_12_evals/predictions_winogrande_dev_in_training.lst'
    eval_dir = r'/cs/labs/roys/aviadsa/cartography/outputs/WINOGRANDE/deberta-large/huji_WINOGRANDE_deberta-large_12_eval_dynamics/eval_dynamics'

    samples_metrics = compute_eval_dynamics(eval_dir)

    with open(data_file1, 'r') as data_file1:
        lines1 = [json.loads(line) for line in data_file1]
    with open(data_file2, 'r') as data_file2:
        lines2 = [json.loads(line) for line in data_file2]

    correct1 = 0
    correct2 = 0
    correct1_addition = 0
    correct2_addition = 0
    correct1_subtraction = 0
    correct2_subtraction = 0
    differ = 0
    differ_variabilities = []
    differ_confidences = []
    differ_final_confidences = []
    agree_variabilities = []
    agree_confidences = []
    agree_final_confidences = []
    for line1, line2, samples_metric in zip(lines1, lines2, samples_metrics):
        assert line1['guid'] == line2['guid'] == samples_metric['guid']
        if line1['label'] == line1['gold']:
            correct1 += 1
        if line2['label'] == line2['gold']:
            correct2 += 1
        if (line1['label'] == line1['gold'] and line2['label'] != line2['gold']) or (line1['label'] != line1['gold'] and line2['label'] == line2['gold']):
            differ += 1
            differ_variabilities.append(samples_metric['variability'])
            differ_confidences.append(samples_metric['overall_confidence'])
            differ_final_confidences.append(samples_metric['final_confidence'])
            # print('confidences of differ sample: {}, {}'.format(line1['confidence'], line2['confidence']))
            if samples_metric['variability'] >= 0.145 and line2['label'] == line2['gold']:
                correct1_addition += 1
            elif samples_metric['variability'] >= 0.145 and line1['label'] == line1['gold']:
                correct1_subtraction += 1
            elif samples_metric['variability'] < 0.145 and line1['label'] == line1['gold']:
                correct2_addition += 1
            elif samples_metric['variability'] < 0.145 and line2['label'] == line2['gold']:
                correct2_subtraction += 1

            # if line1['confidence'] >= line2['confidence'] and line1['label'] == line1['gold']:
            #     correct2_addition += 1
            # elif line1['confidence'] < line2['confidence'] and line2['label'] == line2['gold']:
            #     correct1_addition += 1
            # elif line1['confidence'] >= line2['confidence'] and line1['label'] != line1['gold']:
            #     correct1_subtraction += 1
            # elif line1['confidence'] < line2['confidence'] and line2['label'] != line2['gold']:
            #     correct2_subtraction += 1
        else:
            agree_variabilities.append(samples_metric['variability'])
            agree_confidences.append(samples_metric['overall_confidence'])
            agree_final_confidences.append(samples_metric['final_confidence'])

    print(differ_variabilities[:20])
    print(agree_variabilities[:20])
    print('num samples: {}'.format(len(lines1)))
    print('correct prediction in 1st model: {}'.format(correct1))
    print('correct prediction in 2st model: {}'.format(correct2))
    print('correct addition in 1st model: {}'.format(correct1_addition))
    print('correct addition in 2ed model: {}'.format(correct2_addition))
    print('correct subtraction in 1st model: {}'.format(correct1_subtraction))
    print('correct subtraction in 2ed model: {}'.format(correct2_subtraction))
    print('correct prediction difference: {}'.format(correct2 - correct1))
    print('num individual predictions that differ: {}'.format(differ))
    print('mean differ variability: {}'.format(np.mean(differ_variabilities)))
    print('mean agree variability: {}'.format(np.mean(agree_variabilities)))
    print('min differ variability: {}'.format(np.min(differ_variabilities)))
    print('max agree variability: {}'.format(np.max(agree_variabilities)))
    print('mean differ confidence: {}'.format(np.mean(differ_confidences)))
    print('mean agree confidence: {}'.format(np.mean(agree_confidences)))


def compute_eval_dynamics(eval_dir):
    variability_func = lambda conf: np.std(conf)

    contents = []
    for filename in os.listdir(eval_dir):
        with open(os.path.join(eval_dir, filename), 'r') as file:
            content = [json.loads(line) for line in file]
            contents.append(content)

    samples = []
    for sample in zip(*contents):
        for idx, line in enumerate(sample):
            if idx == 0:
                line['probabilities'] = [line['probabilities']]
                samples.append(line)
            else:
                samples[-1]['probabilities'].append(line['probabilities'])

    for sample in samples:
        gold = int(sample['gold']) - 1
        true_probs_trend = []
        for probs in sample['probabilities']:
            true_class_prob = float(probs[gold])
            true_probs_trend.append(true_class_prob)

        sample['final_confidence'] = true_probs_trend[-1]
        sample['overall_confidence'] = np.mean(true_probs_trend)
        sample['variability'] = variability_func(true_probs_trend)

    return samples


def compare_variability_confindence():
    eval_dir = r'/cs/labs/roys/aviadsa/cartography/outputs/WINOGRANDE/deberta-large/huji_WINOGRANDE_deberta-large_12_eval_dynamics/eval_dynamics'
    samples_metrics = compute_eval_dynamics(eval_dir)

    def compare_confidence(x, y):
        if abs(x['overall_confidence'] - 0.5) < abs(y['overall_confidence'] - 0.5):
            return -1
        elif abs(y['overall_confidence'] - 0.5) < abs(x['overall_confidence'] - 0.5):
            return 1
        else:
            return 0

    samples_metrics_confidence = sorted(samples_metrics, key=functools.cmp_to_key(compare_confidence))
    samples_metrics_variability = sorted(samples_metrics, key=lambda s: s['variability'], reverse=True)

    mid_50_confidence = samples_metrics_confidence[:len(samples_metrics) // 2]
    top_50_variability = samples_metrics_variability[:len(samples_metrics) // 2]

    mid_50_confidence_ids = [sample['guid'] for sample in mid_50_confidence]
    top_50_variability_ids = [sample['guid'] for sample in top_50_variability]

    print(len(set(mid_50_confidence_ids)))
    print(len(set(top_50_variability_ids)))
    print(len(set(mid_50_confidence_ids).intersection(set(top_50_variability_ids))))


compare_correct_samples()

# compare_variability_confindence()


