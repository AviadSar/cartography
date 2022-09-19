import os
import json

import numpy as np

file_name = 'eval_metrics_train.json'
json_outputs = {}

for root, dirs, files in os.walk("/cs/labs/roys/aviadsa/cartography/outputs"):
    dirs.sort()
    if file_name in files:
        root_dirs = root.split('/')
        # for collecting bias experiments
        if '10_epochs' in root_dirs[-1] or '15_epochs' in root_dirs[-1] or 'mix' in root_dirs[-1]\
                or '_bert' in root_dirs[-1] or 'xlnet' in root_dirs[-1] or 't5' in root_dirs[-1]\
                or '_epochs' in root_dirs[-1] or ('12_evals' not in root_dirs[-1]) or 'abductive_nli' in root_dirs[-1]\
                or 'anli_v1.0_R1' in root_dirs[-1] or 'roberta-large_128_batch_12_evals' in root_dirs[-1]:
            continue
        # if '24_evals' not in root_dirs[-1]:
        #     continue
        with open(os.path.join(root, file_name), 'r') as file:
            # json_lines = [json.loads(line) for line in file]
            json_lines = file.readlines()

        if root_dirs[-3] not in json_outputs:
            json_outputs[root_dirs[-3]] = {}
        if root_dirs[-2] not in json_outputs[root_dirs[-3]]:
            json_outputs[root_dirs[-3]][root_dirs[-2]] = {}

        json_outputs[root_dirs[-3]][root_dirs[-2]][root_dirs[-1]] = json_lines

with open("/cs/labs/roys/aviadsa/cartography/outputs/collected_outputs.json", 'w') as collected_outputs:
    json.dump(json_outputs, collected_outputs, indent=4)

eval_cycle = -1
task_names = []
task_model_names = []
results = []
for task_name in json_outputs:
    task = json_outputs[task_name]
    model_names = []
    task_results = []
    for model_name in task:
        model = task[model_name]
        whole_set_result, random_result = None, None
        grid_result = [[None] * 3, [None] * 3, [None] * 3]
        for exp_name in model:
            lines = model[exp_name]
            if 'variability' not in exp_name and 'random' not in exp_name:
                whole_set_result = json.loads(lines[eval_cycle])['best_dev_performance']
            elif 'random' in exp_name:
                random_result = json.loads(lines[eval_cycle])['best_dev_performance']
            else:
                for bias_idx, bias in enumerate(['bias_2', 'bias_3', 'bias_4']):
                    for percent_idx, percent in enumerate(['0.25', '0.33', '0.50']):
                        if bias in exp_name and percent in exp_name:
                            # print(bias_idx, percent_idx, exp_name, json.loads(lines[-1])['best_dev_performance'])
                            grid_result[bias_idx][percent_idx] = json.loads(lines[eval_cycle])['best_dev_performance']
        model_reults = [random_result - whole_set_result, [[None] * 3, [None] * 3, [None] * 3]]
        for bias_idx in [0, 1, 2]:
            for percent_idx in [0, 1, 2]:
                model_reults[1][bias_idx][percent_idx] = grid_result[bias_idx][percent_idx] - whole_set_result
        model_names.append(model_name)
        task_results.append(model_reults)
    task_names.append(task_name)
    task_model_names.append(model_names)
    results.append(task_results)

num_models = 0
random_results = 0
grid_resluts = [[0] * 3, [0] * 3, [0] * 3]
for task_idx, task_name in enumerate(task_names):
    if task_idx == 1:
        print(task_name)
    else:
        continue
    for model_idx, model_name in enumerate(task_model_names[task_idx]):
        num_models += 1
        random_results += results[task_idx][model_idx][0]
        for bias_idx, bias in enumerate(['bias_2', 'bias_3', 'bias_4']):
            for percent_idx, percent in enumerate(['0.25', '0.33', '0.50']):
                grid_resluts[bias_idx][percent_idx] += results[task_idx][model_idx][1][bias_idx][percent_idx]
                # if bias == 'bias_3' and percent == '0.50':
                #     print(task_name, model_name, results[task_idx][model_idx][1][bias_idx][percent_idx])
for bias_idx, bias in enumerate(['bias_2', 'bias_3', 'bias_4']):
    for percent_idx, percent in enumerate(['0.25', '0.33', '0.50']):
        grid_resluts[bias_idx][percent_idx] = grid_resluts[bias_idx][percent_idx] / num_models
random_results /= num_models

print(grid_resluts)
print(random_results)
