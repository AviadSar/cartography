import os
import json

import numpy as np

file_name = 'eval_metrics_train.json'
json_outputs = {}

for root, dirs, files in os.walk("/cs/labs/roys/aviadsa/cartography/outputs"):
    dirs.sort()
    if file_name in files:
        root_dirs = root.split('/')
        if '10_epochs' in root_dirs[-1] or '15_epochs' in root_dirs[-1] or 'mix' in root_dirs[-1]\
                or '_bert' in root_dirs[-1]  or 'xlnet' in root_dirs[-1] or 't5' in root_dirs[-1]:
            continue
        with open(os.path.join(root, file_name), 'r') as file:
            # json_lines = [json.loads(line) for line in file]
            json_lines = file.readlines()

        if root_dirs[-3] not in json_outputs:
            json_outputs[root_dirs[-3]] = {}
        if root_dirs[-2] not in json_outputs[root_dirs[-3]]:
            json_outputs[root_dirs[-3]][root_dirs[-2]] = {}

        json_outputs[root_dirs[-3]][root_dirs[-2]][root_dirs[-1]] = json_lines

results_dict = {}
results = None
for task_name in json_outputs:
    task = json_outputs[task_name]
    task_results_dict = {}
    task_results_strs = {}
    task_results = None
    for model_name in task:
        model = json_outputs[model_name]
        whole_set_result, bias_2_result = None, None
        for exp_name in model:
            lines = model[exp_name]
            if 'variability' not in exp_name and 'random' not in exp_name:
                whole_set_result = json.loads(lines[-1])['best_dev_performance']
            if 'bias_2' in exp_name:
                if len(lines) > 3:
                    bias_2_result = json.loads(lines[3])['best_dev_performance']
                    bias_2_epoch = 4
                else:
                    bias_2_result = json.loads(lines[-1])['best_dev_performance']
                    bias_2_epoch = len(lines) + 1
        if whole_set_result is None or bias_2_result is None:
            model_results = 'no results. whole set result is {}, and bias 2 result is {}'.format(whole_set_result, bias_2_result)
            continue
        else:
            model_results = '{}, on epoch {}'.format(bias_2_result - whole_set_result, bias_2_epoch)
        task_results_strs[model_name] =  model_results
        task_results_dict[model_name] = bias_2_result - whole_set_result
    task_results = np.mean(task_results_dict.values())
    results_dict[task_name] = task_results

    print(task_name + ': ')
    print(task_results_strs)

results = np.mean(results_dict.values())
print(results_dict)
print('results: {}'.format(results))





with open("/cs/labs/roys/aviadsa/cartography/outputs/collected_outputs.json", 'w') as collected_outputs:
    json.dump(json_outputs, collected_outputs, indent=4)
