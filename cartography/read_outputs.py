import os
import json

import numpy as np

file_name = 'eval_metrics_train.json'
json_outputs = {}

# reading winogrande td results
# exp_names = ['huji_WINOGRANDE_deberta-large_128_batch_24_evals_seed',
#              "bias_2_confidence_0.50_dt__strat_dt_epoch_1",
#              "bias_2_confidence_0.50_dt__strat_dt_epoch_2",
#              "bias_2_confidence_0.50_dt__strat_dt_epoch_3",
#              "bias_2_variability_0.50_dt__strat_dt_epoch_1",
#              "bias_2_variability_0.50_dt__strat_dt_epoch_2",
#              "bias_2_variability_0.50_dt__strat_dt_epoch_3",
#              "bias_3_confidence_0.50_dt__strat_dt_epoch_1",
#              "bias_3_confidence_0.50_dt__strat_dt_epoch_2",
#              "bias_3_confidence_0.50_dt__strat_dt_epoch_3",
#              "bias_3_variability_0.50_dt__strat_dt_epoch_1",
#              "bias_3_variability_0.50_dt__strat_dt_epoch_2",
#              "bias_3_variability_0.50_dt__strat_dt_epoch_3",
#              ]

# reading SNLI, WINOGRANDE, anli, and WINOGRANDE td results
exp_names = ['huji_WINOGRANDE_deberta-large_128_batch_24_evals_seed',
             "bias_2_confidence_0.50_dt__strat_dt_epoch_1",
             "bias_2_confidence_0.50_dt__strat_dt_epoch_2",
             "bias_2_confidence_0.50_dt__strat_dt_epoch_3",
             "bias_2_variability_0.50_dt__strat_dt_epoch_1",
             "bias_2_variability_0.50_dt__strat_dt_epoch_2",
             "bias_2_variability_0.50_dt__strat_dt_epoch_3",
             "bias_3_confidence_0.50_dt__strat_dt_epoch_1",
             "bias_3_confidence_0.50_dt__strat_dt_epoch_2",
             "bias_3_confidence_0.50_dt__strat_dt_epoch_3",
             "bias_3_variability_0.50_dt__strat_dt_epoch_1",
             "bias_3_variability_0.50_dt__strat_dt_epoch_2",
             "bias_3_variability_0.50_dt__strat_dt_epoch_3",
             'huji_SNLI_deberta-large_128_batch_12_evals',
             'huji_SNLI_deberta-large_on_deberta-large_variability_0.25_bias_2_12_evals',
             'huji_SNLI_deberta-large_on_deberta-large_variability_0.25_bias_3_12_evals',
             'huji_SNLI_deberta-large_on_deberta-large_variability_0.25_bias_4_12_evals',
             'huji_SNLI_deberta-large_on_deberta-large_variability_0.33_bias_2_12_evals',
             'huji_SNLI_deberta-large_on_deberta-large_variability_0.33_bias_3_12_evals',
             'huji_SNLI_deberta-large_on_deberta-large_variability_0.33_bias_4_12_evals',
             'huji_SNLI_deberta-large_on_deberta-large_variability_0.50_bias_2_12_evals',
             'huji_SNLI_deberta-large_on_deberta-large_variability_0.50_bias_3_12_evals',
             'huji_SNLI_deberta-large_on_deberta-large_variability_0.50_bias_4_12_evals',
             'huji_WINOGRANDE_deberta-large_128_batch_12_evals',
             'huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.25_bias_2_12_evals',
             'huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.25_bias_3_12_evals',
             'huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.25_bias_4_12_evals',
             'huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.33_bias_2_12_evals',
             'huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.33_bias_3_12_evals',
             'huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.33_bias_4_12_evals',
             'huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.50_bias_2_12_evals',
             'huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.50_bias_3_12_evals',
             'huji_WINOGRANDE_deberta-large_on_deberta-large_variability_0.50_bias_4_12_evals',
             'huji_anli_v1.0_R3_deberta-large_128_batch_12_evals',
             'huji_anli_v1.0_R3_deberta-large_on_deberta-large_variability_0.25_bias_2_12_evals',
             'huji_anli_v1.0_R3_deberta-large_on_deberta-large_variability_0.25_bias_3_12_evals',
             'huji_anli_v1.0_R3_deberta-large_on_deberta-large_variability_0.25_bias_4_12_evals',
             'huji_anli_v1.0_R3_deberta-large_on_deberta-large_variability_0.33_bias_2_12_evals',
             'huji_anli_v1.0_R3_deberta-large_on_deberta-large_variability_0.33_bias_3_12_evals',
             'huji_anli_v1.0_R3_deberta-large_on_deberta-large_variability_0.33_bias_4_12_evals',
             'huji_anli_v1.0_R3_deberta-large_on_deberta-large_variability_0.50_bias_2_12_evals',
             'huji_anli_v1.0_R3_deberta-large_on_deberta-large_variability_0.50_bias_3_12_evals',
             'huji_anli_v1.0_R3_deberta-large_on_deberta-large_variability_0.50_bias_4_12_evals',
             'huji_SNLI_electra-large_128_batch_12_evals',
             'huji_SNLI_electra-large_on_electra-large_variability_0.25_bias_2_12_evals',
             'huji_SNLI_electra-large_on_electra-large_variability_0.25_bias_3_12_evals',
             'huji_SNLI_electra-large_on_electra-large_variability_0.25_bias_4_12_evals',
             'huji_SNLI_electra-large_on_electra-large_variability_0.33_bias_2_12_evals',
             'huji_SNLI_electra-large_on_electra-large_variability_0.33_bias_3_12_evals',
             'huji_SNLI_electra-large_on_electra-large_variability_0.33_bias_4_12_evals',
             'huji_SNLI_electra-large_on_electra-large_variability_0.50_bias_2_12_evals',
             'huji_SNLI_electra-large_on_electra-large_variability_0.50_bias_3_12_evals',
             'huji_SNLI_electra-large_on_electra-large_variability_0.50_bias_4_12_evals',
             'huji_WINOGRANDE_electra-large_128_batch_12_evals',
             'huji_WINOGRANDE_electra-large_on_electra-large_variability_0.25_bias_2_12_evals',
             'huji_WINOGRANDE_electra-large_on_electra-large_variability_0.25_bias_3_12_evals',
             'huji_WINOGRANDE_electra-large_on_electra-large_variability_0.25_bias_4_12_evals',
             'huji_WINOGRANDE_electra-large_on_electra-large_variability_0.33_bias_2_12_evals',
             'huji_WINOGRANDE_electra-large_on_electra-large_variability_0.33_bias_3_12_evals',
             'huji_WINOGRANDE_electra-large_on_electra-large_variability_0.33_bias_4_12_evals',
             'huji_WINOGRANDE_electra-large_on_electra-large_variability_0.50_bias_2_12_evals',
             'huji_WINOGRANDE_electra-large_on_electra-large_variability_0.50_bias_3_12_evals',
             'huji_WINOGRANDE_electra-large_on_electra-large_variability_0.50_bias_4_12_evals',
             'huji_anli_v1.0_R3_electra-large_128_batch_12_evals',
             'huji_anli_v1.0_R3_electra-large_on_electra-large_variability_0.25_bias_2_12_evals',
             'huji_anli_v1.0_R3_electra-large_on_electra-large_variability_0.25_bias_3_12_evals',
             'huji_anli_v1.0_R3_electra-large_on_electra-large_variability_0.25_bias_4_12_evals',
             'huji_anli_v1.0_R3_electra-large_on_electra-large_variability_0.33_bias_2_12_evals',
             'huji_anli_v1.0_R3_electra-large_on_electra-large_variability_0.33_bias_3_12_evals',
             'huji_anli_v1.0_R3_electra-large_on_electra-large_variability_0.33_bias_4_12_evals',
             'huji_anli_v1.0_R3_electra-large_on_electra-large_variability_0.50_bias_2_12_evals',
             'huji_anli_v1.0_R3_electra-large_on_electra-large_variability_0.50_bias_3_12_evals',
             'huji_anli_v1.0_R3_electra-large_on_electra-large_variability_0.50_bias_4_12_evals',
             ]
exp_vals = [[0, 0] for exp_name in exp_names]
# print(exp_vals)
for root, dirs, files in os.walk("/cs/labs/roys/aviadsa/cartography/outputs/"):
    dirs.sort()
    if file_name in files:
        root_dirs = root.split('/')
        for i, exp_name in enumerate(exp_names):
            if exp_name in root_dirs[-1]:
                with open(os.path.join(root, file_name), 'r') as file:
                    json_lines = [json.loads(line) for line in file]
                    exp_vals[i][0] += 1
                    exp_vals[i][1] += json_lines[-1]['best_dev_performance']

for i, exp_val in enumerate(exp_vals):
    # if exp_val[0] != 1:
        # print('{} exps found for exp name {}, instead of 1'.format(exp_val[0], exp_names[i]))
    if exp_val[0] != 0:
        print('{}: {}'.format(exp_names[i], float(exp_val[1]) / exp_val[0]))
    else:
        print('WARNING! division by zero at exp name {}'.format(exp_names[i]))


