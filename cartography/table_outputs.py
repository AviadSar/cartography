import pandas as pd
import os
import json

'huji_anli_v1.0_R3_deberta-large_on_deberta-large_bias_2_variability_0.50_dt_strat_dt_epoch_0_128_batch_24_evals_seed_42'


def mean_and_std_results(results):
    for key, result_list in results.items():
        summed_listings = []
        for single_seed_results in result_list:
            for idx, single_eval_result in enumerate(single_seed_results):
                if idx >= len(summed_listings):
                    single_eval_result_for_summing = {}
                    for key, value in single_eval_result.items():
                        single_eval_result_for_summing[key] = [value]
                    summed_listings.append(single_eval_result_for_summing)
                else:
                    for key, value in single_eval_result.items():
                        summed_listings[idx][key].append(value)


with open('/cs/labs/roys/aviadsa/cartography/outputs/experimets', 'r') as experimets_file:
    experimets = [json.loads(line) for line in experimets_file.readlines()]

experimets.sort(key=lambda experiment: experimet['file_name'])

summed_experiments = []
file_name_no_seed = experimets[0]['file_name'][:-2]
results = {}
for idx, experiment in enumerate(experimets):
    if os.path.exists(experiment['directory_name']):
        for file_or_directory in os.listdir(experiment['directory_name']):
            if 'eval_metrics' in file_or_directory:
                with open(os.path.join(experiment['directory_name'], file_or_directory), 'r') as fod:
                    if file_or_directory.split['.'][0] in results:
                        results[file_or_directory].append([json.loads(line) for line in fod])
                    else:
                        results[file_or_directory] = [[json.loads(line) for line in fod]]
    if (idx + 1) >= len(experimets) or experimets[idx + 1]['file_name'][:-2] != file_name_no_seed:
        results = mean_and_std_results(results)
        experiment.update(results)
        summed_experiments.append(experiment)
        results = {}