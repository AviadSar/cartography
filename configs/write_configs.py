import os
import json

config_string = \
'local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_VISIBLE_DEVICES"));\n\
\n\
local LEARNING_RATE = {0};\n\
local BATCH_SIZE = {1};\n\
local NUM_EPOCHS = {2};\n\
local SEED = {3};\n\
\n\
local TASK = "{4}";\n\
local DATA_DIR = "{5}";\n\
local FEATURES_CACHE_DIR = "{6}";\n\
\n\
local TEST = "{7}";\n\
\n\
{{\n\
    "data_dir": DATA_DIR,\n\
    "model_type": "{8}",\n\
    "model_name_or_path": "{9}",\n\
    "data_model_name_or_path": "{10}",\n\
    "task_name": TASK,\n\
    "seed": SEED,\n\
    "num_train_epochs": NUM_EPOCHS,\n\
    "learning_rate": LEARNING_RATE,\n\
    "features_cache_dir": FEATURES_CACHE_DIR,\n\
    "per_gpu_train_batch_size": BATCH_SIZE,\n\
    "per_gpu_eval_batch_size": BATCH_SIZE,\n\
    "gradient_accumulation_steps": {11},\n\
    "do_train": true,\n\
    "do_eval": true,\n\
    "do_test": true,\n\
    "test": TEST,\n\
    "patience": {12},\n\
    "train_set_fraction": {13},\n\
    "max_seq_length": {14},\n\
    "eval_steps": {15},\n\
    "save_steps": {16},\n\
    "num_eval_cycles": {17},\n\
    "granularity": {18},\n\
    "metric": "{19}",\n\
    "bias": {20},\n\
    "favored_fraction": {21},\n\
    "start_dt_epoch": {22},\n\
    "td_dir": "{23}",\n\
    "overwrite_output_dir": true,\n\
    "model_weights_output_dir": "{24}",\n\
    "save_model": {25},\n\
    "eval_tasks_names": {26},\n\
    "burn_out": {27},\n\
    "burn_in": {28},\n\
    "reboot_on_epoch": {29},\n\
    "from_reboot": {30},\n\
    "extract": {31},\n\
    "extract_threshold": {32},\n\
    "extract_patience": {33},\n\
    "from_extract": {34}, \n\
    "from_extract_threshold": {35}, \n\
    "from_extract_patience": {36}, \n\
    "mix_confidence": {37}, \n\
}}'

slurm_config_string = \
'#!/bin/bash\n\
#SBATCH --mem={0}\n\
#SBATCH --gres={1}\n\
#SBATCH --{2}\n\
#SBATCH --output=/cs/labs/roys/aviadsa/cartography/slurm_out_files/{3}.txt\n\
#SBATCH --killable\n\
\n\
python cartography/classification/run_glue.py -c configs/{3}.jsonnet --do_train -o outputs/{4}/\n'

unkillable_slurm_config_string = \
'#!/bin/bash\n\
#SBATCH --mem={0}\n\
#SBATCH --gres={1}\n\
#SBATCH --{2}\n\
#SBATCH --output=/cs/labs/roys/aviadsa/cartography/slurm_out_files/{3}.txt\n\
\n\
python cartography/classification/run_glue.py -c configs/{3}.jsonnet --do_train -o outputs/{4}/\n'

slurm_restricted_string = \
'python cartography/classification/run_glue.py -c configs/{}.jsonnet --do_train -o outputs/{}/\n'

# filter_string = \
# 'python cartography/selection/train_dy_filtering.py --plot --filter --task_name {0} --metric variability --model {1} --model_dir outputs/{2} --data_dir /cs/labs/roys/aviadsa/datasets/cartography/{3}  --plots_dir /cs/labs/roys/aviadsa/cartography/cartography/plots --filtering_output_dir /cs/labs/roys/aviadsa/datasets/cartography/filtered_datasets\n'

filter_string = \
'python cartography/selection/train_dy_filtering.py --plot --filter --task_name {0} --metric variability --model {1} --model_dir outputs/{2} --data_dir /cs/labs/roys/aviadsa/datasets/cartography/{3}  --plots_dir /cs/labs/roys/aviadsa/cartography/cartography/plots --filtering_output_dir /cs/labs/roys/aviadsa/datasets/cartography/filtered_datasets\n\
python cartography/selection/train_dy_filtering.py --filter --task_name {0} --metric random --model {1} --model_dir outputs/{2} --data_dir /cs/labs/roys/aviadsa/datasets/cartography/{3}  --plots_dir /cs/labs/roys/aviadsa/cartography/cartography/plots --filtering_output_dir /cs/labs/roys/aviadsa/datasets/cartography/filtered_datasets\n'

run_string = 'sbatch {0} >> slurm_configs/submitted_jobs.txt && truncate -s-1 slurm_configs/submitted_jobs.txt && echo -n " sbatch {0}\\n" >> slurm_configs/submitted_jobs.txt\n'


def write_config(workspace,
                 file_name,
                 learning_rate,
                 batch_size,
                 num_epochs,
                 seed,
                 task,
                 data_dir,
                 cache_dir,
                 test_dir,
                 model_name,
                 data_model_name,
                 model_type,
                 model_name_or_path,
                 data_model_name_or_path,
                 gradient_accumulation_steps,
                 data_gradient_accumulation,
                 data_seed,
                 patience,
                 mem,
                 gres,
                 time,
                 train_set_fraction,
                 max_seq_length,
                 eval_steps,
                 save_steps,
                 num_eval_cycles,
                 num_data_eval_cycles,
                 granularity,
                 metric,
                 bias,
                 favored_fraction,
                 start_dt_epoch,
                 save_model,
                 eval_tasks_names,
                 burn_out,
                 burn_in,
                 reboot_on_epoch,
                 from_reboot,
                 extract,
                 extract_threshold,
                 extract_patience,
                 from_extract,
                 from_extract_threshold,
                 from_extract_patience,
                 mix_confidence,
                 unkillable
                 ):
    config_dir = os.path.join('/cs/labs/roys/aviadsa/cartography/configs', task, model_name)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    if data_model_name is not None:
        td_dir = os.path.join('outputs', task, data_model_name,
                              '{}_{}_{}{}{}_{}_batch_{}_evals_seed_{}'.format(workspace,
                                                                            task,
                                                                            data_model_name,
                                                                            '_reboot' if from_reboot else '',
                                                                            '_extract_{}_{}'.format(from_extract_threshold, from_extract_patience) if from_extract else '',
                                                                            data_gradient_accumulation,
                                                                            num_data_eval_cycles,
                                                                            data_seed))
    else:
        td_dir = os.path.join('outputs', task, model_name, file_name)
    model_weights_output_dir = os.path.join('/cs/snapless/roys/aviadsa/cartography/outputs', task, model_name, file_name)

    with open(os.path.join(config_dir, file_name + '.jsonnet'), 'w') as config_file:
        config_file.write(config_string.format(
            learning_rate,
            batch_size,
            num_epochs,
            seed,
            task,
            data_dir,
            cache_dir,
            test_dir,
            model_type,
            model_name_or_path,
            data_model_name_or_path,
            gradient_accumulation_steps,
            patience,
            train_set_fraction,
            max_seq_length,
            eval_steps,
            save_steps,
            num_eval_cycles,
            granularity if granularity is not None else 'null',
            metric if metric is not None else 'null',
            bias if bias is not None else 'null',
            favored_fraction if favored_fraction is not None else 'null',
            start_dt_epoch if start_dt_epoch is not None else 'null',
            td_dir,
            model_weights_output_dir,
            save_model,
            eval_tasks_names,
            burn_out if burn_out is not None else 'null',
            burn_in if burn_in is not None else 'null',
            'true' if reboot_on_epoch else 'false',
            'true' if from_reboot else 'false',
            'true' if extract else 'false',
            extract_threshold,
            extract_patience,
            'true' if from_extract else 'false',
            from_extract_threshold,
            from_extract_patience,
            mix_confidence if mix_confidence is not None else 'null',
        ))

    slurm_dir = os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_configs', task, model_name)
    if not os.path.exists(slurm_dir):
        os.makedirs(slurm_dir)

    slurm_out_dir = os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_out_files', task, model_name)
    if not os.path.exists(slurm_out_dir):
        os.makedirs(slurm_out_dir)

    with open(os.path.join(slurm_dir, file_name + '.sh'), 'w') as slurm_file:
        acting_slurm_config_string = unkillable_slurm_config_string if unkillable else slurm_config_string
        slurm_file.write(acting_slurm_config_string.format(
            mem,
            gres,
            time,
            os.path.join(task, model_name, file_name),
            os.path.join(task, model_name, file_name)
        ))
        # if data_model_name_or_path == '' and favored_fraction is None:
        #     slurm_file.write(filter_string.format(
        #         task,
        #         model_name,
        #         os.path.join(task, model_name, file_name),
        #         task
        #     ))

    consecutive = False
    if consecutive:
        with open(os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_configs', 'run_consecutive.sh'),
                  'a') as run_file:
            run_file.write(slurm_restricted_string.format(
                os.path.join(task, model_name, file_name),
                os.path.join(task, model_name, file_name)
            ))
            # if data_model_name_or_path == '' and favored_fraction is None:
            #     run_file.write(filter_string.format(
            #         task,
            #         model_name,
            #         os.path.join(task, model_name, file_name),
            #         task
            #     ))

    with open(os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_configs/run_slurm.sh'), 'a') as run_file:
        run_file.write(run_string.format(os.path.join(slurm_dir, file_name + '.sh')))

    with open('/cs/labs/roys/aviadsa/cartography/outputs/experiments.jsonl', 'r+') as experiments_file:
        experiments = [json.loads(line) for line in experiments_file.readlines()]
        directory_names = [experiment['directory_name'] for experiment in experiments]
        directory_name = os.path.join('/cs/labs/roys/aviadsa/cartography/outputs', task, model_name, file_name)

        experiment = {'directory_name': directory_name,
                      'workspace': workspace,
                      'file_name': file_name,
                      'learning_rate': learning_rate,
                      'batch_size': batch_size,
                      'num_epochs': num_epochs,
                      'seed': seed,
                      'task': task,
                      'data_dir': data_dir,
                      'cache_dir': cache_dir,
                      'test_dir': test_dir,
                      'model_name': model_name,
                      'data_model_name': data_model_name,
                      'model_type': model_type,
                      'model_name_or_path': model_name_or_path,
                      'data_model_name_or_path': data_model_name_or_path,
                      'gradient_accumulation_steps': gradient_accumulation_steps,
                      'data_gradient_accumulation': data_gradient_accumulation,
                      'data_seed': data_seed,
                      'patience': patience,
                      'mem': mem,
                      'gres': gres,
                      'time': time,
                      'train_set_fraction': train_set_fraction,
                      'max_seq_length': max_seq_length,
                      'eval_steps': eval_steps,
                      'save_steps': save_steps,
                      'num_eval_cycles': num_eval_cycles,
                      'num_data_eval_cycles': num_data_eval_cycles,
                      'granularity': granularity,
                      'metric': metric,
                      'bias': bias,
                      'favored_fraction': favored_fraction,
                      'start_dt_epoch': start_dt_epoch,
                      'save_model': save_model,
                      'eval_tasks_names': eval_tasks_names,
                      'burn_out': burn_out,
                      'burn_in': burn_in,
                      'reboot_on_epoch': reboot_on_epoch,
                      'from_reboot': from_reboot,
                      'extract': extract,
                      'extract_threshold': extract_threshold,
                      'extract_patience': extract_patience,
                      'from_extract': from_extract,
                      'from_extract_threshold': from_extract_threshold,
                      'from_extract_patience': from_extract_patience,
                      'mix_confidence': mix_confidence,
                      'unkillable':unkillable
                      }
        if directory_name not in directory_names:
            experiments_file.write(json.dumps(experiment) + '\n')


def quick_write_config(workspace, task, model, model_type, data_dir_suffix='', cache_dir_suffix='', filtering='',
                       data_model='', model_name=None, data_model_name=None, seed='42', batch_size='4', num_epochs='5',
                       num_data_epochs='5', mem='16g', gres='gpu:1,vmem:10g', time='time=48:0:0',
                       train_set_fraction='1.0', max_seq_length='128', bias=None, gradient_accumulation='128',
                       data_gradient_accumulation='128', data_seed='42', eval_samples='12800', save_steps='12800',
                       num_eval_cycles='24', num_data_eval_cycles='24', granularity=None, metric=None,
                       favored_fraction=None, start_dt_epoch=None, patience='100', save_model='true',
                       eval_tasks_names=[], burn_out=None, burn_in=None, reboot_on_epoch=False, from_reboot=False,
                       extract=False, extract_threshold='1.05', extract_patience='2',
                       from_extract=False, from_extract_threshold='1.05', from_extract_patience='2', mix_confidence=None,
                       unkillable=False):
    if model_name is None:
        model_name = model
    if data_model_name is None:
        data_model_name = data_model

    if workspace == 'home':
        data_dir_prefix = r'C:\my_documents\datasets\cartography\\'
        cache_dir_prefix = r'C:\my_documents\datasets\cartography\\'
    elif workspace == 'huji':
        data_dir_prefix = r'/cs/labs/roys/aviadsa/datasets/cartography/'
        cache_dir_prefix = r'/cs/labs/roys/aviadsa/datasets/cartography/'
    else:
        raise ValueError('no such workspace {}'.format(workspace))

    file_name = '{}{}_{}_{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}_{}_batch_{}_evals_seed_{}'.format(
                                                'unkillable_' if unkillable else '',
                                                workspace, task, model_name,
                                                '_on_' + data_model_name if data_model_name != '' else '',
                                                '_' + filtering if filtering != '' else '',
                                                '_bias_' + bias if bias is not None else '',
                                                '_' + metric if metric is not None else '',
                                                '_' + favored_fraction + '_dt' if favored_fraction is not None else '',
                                                '_mix_' + mix_confidence if mix_confidence is not None else '',
                                                '_strat_dt_epoch_' + start_dt_epoch if start_dt_epoch is not None else '',
                                                '_gran_' + granularity if granularity is not None else '',
                                                '_burn_out_' + burn_out if burn_out is not None else '',
                                                '_burn_in_' + burn_in if burn_in is not None else '',
                                                '_reboot' if reboot_on_epoch else '',
                                                '_from_reboot' if from_reboot else '',
                                                '_extract_{}_{}'.format(extract_threshold, extract_patience) if extract else '',
                                                '_from_extract_{}_{}'.format(from_extract_threshold, from_extract_patience) if from_extract else '',
                                                gradient_accumulation,
                                                num_eval_cycles,
                                                seed)
    if filtering != '':
        data_dir = os.path.join(data_dir_prefix, 'filtered_datasets',
                                'cartography_' + filtering + ('_bias_' + bias if bias is not None else ''), task,
                                '{}_{}_{}_{}_evals'.format(workspace, task, data_model_name, num_data_eval_cycles))
    else:
        data_dir = os.path.join(data_dir_prefix, task)
    cache_dir = os.path.join(cache_dir_prefix, task, 'cache_{}{}{}{}{}{}{}{}_batch_{}_{}'.format(model_name,
                                                                              '_on_' + data_model_name if data_model_name != '' else '',
                                                                              '_' + filtering if filtering != '' else '',
                                                                              '_bias_' + bias if bias is not None else '',
                                                                              '_' + metric if metric is not None else '',
                                                                              '_' + favored_fraction + '_dt_' if favored_fraction is not None else '',
                                                                              '_mix_' + mix_confidence if mix_confidence is not None else '',
                                                                              '_' + granularity if granularity is not None else '',
                                                                              gradient_accumulation,
                                                                              seed))

    write_config(workspace=workspace,
                 file_name=file_name,
                 learning_rate='1.0708609960508476e-05',
                 batch_size=batch_size,
                 num_epochs=num_epochs,
                 seed=seed,
                 task=task,
                 data_dir=data_dir,
                 cache_dir=cache_dir,
                 test_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE/diagnostic_test.tsv',
                 model_name=model_name,
                 data_model_name=data_model_name,
                 model_type=model_type,
                 model_name_or_path=model,
                 data_model_name_or_path=data_model,
                 gradient_accumulation_steps='{} / BATCH_SIZE'.format(gradient_accumulation),
                 data_gradient_accumulation=data_gradient_accumulation,
                 data_seed=data_seed,
                 patience=patience,
                 mem=mem,
                 gres=gres,
                 time=time,
                 train_set_fraction=train_set_fraction,
                 max_seq_length=max_seq_length,
                 eval_steps='{} / {}'.format(eval_samples, gradient_accumulation),
                 save_steps='{} / {}'.format(save_steps, gradient_accumulation),
                 num_eval_cycles=num_eval_cycles,
                 num_data_eval_cycles=num_data_eval_cycles,
                 granularity=granularity,
                 metric=metric,
                 bias=bias,
                 favored_fraction=favored_fraction,
                 start_dt_epoch=start_dt_epoch,
                 save_model=save_model,
                 eval_tasks_names=eval_tasks_names,
                 burn_out=burn_out,
                 burn_in=burn_in,
                 reboot_on_epoch=reboot_on_epoch,
                 from_reboot=from_reboot,
                 extract=extract,
                 extract_threshold=extract_threshold,
                 extract_patience=extract_patience,
                 from_extract=from_extract,
                 from_extract_threshold=from_extract_threshold,
                 from_extract_patience=from_extract_patience,
                 mix_confidence=mix_confidence,
                 unkillable=unkillable
                 )

# ----- baselines -----
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48', reboot_on_epoch=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48', extract=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='43',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='44',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='45',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='46',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
# 
# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26', reboot_on_epoch=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26', extract=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='43',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='44',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='45',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='201', batch_size='2', seed='46',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='256', batch_size='2', seed='42',
                   eval_tasks_names=['boolq'], reboot_on_epoch=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='256', batch_size='2', seed='42',
                   eval_tasks_names=['boolq'], extract=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='256', batch_size='2', seed='42',
                   eval_tasks_names=['boolq'])
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='256', batch_size='2', seed='43',
                   eval_tasks_names=['boolq'])
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='256', batch_size='2', seed='44',
                   eval_tasks_names=['boolq'])
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='256', batch_size='2', seed='45',
                   eval_tasks_names=['boolq'])
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='256', batch_size='2', seed='46',
                   eval_tasks_names=['boolq'])

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], reboot_on_epoch=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], extract=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='43',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='44',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='45',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='46',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='47',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
# 
# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='100', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], reboot_on_epoch=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='100', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], extract=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='100', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='100', seed='43',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='100', seed='44',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='100', seed='45',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='100', seed='46',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
# 
# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48', reboot_on_epoch=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48', extract=True, gres='gpu:1,vmem:12g')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='43',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='44',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='45',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='null', seed='46',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')


# ----- electra baselines -----
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48', reboot_on_epoch=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='43',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='44',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='45',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='46',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='48')
# 
# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26', reboot_on_epoch=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='42',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='43',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='44',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='45',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='46',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='47',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='201', batch_size='2', seed='48',
                   eval_tasks_names=['SNLI', 'anli_v1.0_R3'], num_eval_cycles='26')

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='256', batch_size='2', seed='42',
                   eval_tasks_names=['boolq'], reboot_on_epoch=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='256', batch_size='2', seed='42',
                   eval_tasks_names=['boolq'])
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='256', batch_size='2', seed='43',
                   eval_tasks_names=['boolq'])
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='256', batch_size='2', seed='44',
                   eval_tasks_names=['boolq'])
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='256', batch_size='2', seed='45',
                   eval_tasks_names=['boolq'])
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', max_seq_length='256', batch_size='2', seed='46',
                   eval_tasks_names=['boolq'])

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], reboot_on_epoch=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='43',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='44',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='45',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='46',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
#                    model_name='electra-large', max_seq_length='null', seed='47',
#                    eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
# 
# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='100', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], reboot_on_epoch=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='100', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='100', seed='43',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='100', seed='44',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='100', seed='45',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='100', seed='46',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
# 
# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48', reboot_on_epoch=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='42',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='43',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='44',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='45',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', max_seq_length='null', seed='46',
                   eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], num_eval_cycles='48')


# ----- training on filtered datasets, based on training dynamics -----
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')




# from reboot
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

                   
                   
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 


# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='47',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)

                   
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='47',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True) 


# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)

                   
                   
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 


# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

                   
                   
                   
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 


# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

                   

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 


# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

                   
                   
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 





# no bias
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48')

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24')


# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26')

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48')





# no bias from reboot
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True)


# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True)

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True)


# no bias mix confidence
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', mix_confidence='0.33')


# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', mix_confidence='0.33')

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], mix_confidence='0.33')

# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', mix_confidence='0.33')





# from reboot no bias mix confidence
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', num_eval_cycles='8', num_data_eval_cycles='24', from_reboot=True, mix_confidence='0.33')


# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='9', num_data_eval_cycles='26', from_reboot=True, mix_confidence='0.33')

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', num_eval_cycles='8', num_data_eval_cycles='24', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, mix_confidence='0.33')

# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias=None, metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='16', num_data_eval_cycles='48', from_reboot=True, mix_confidence='0.33')



# from extract
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_extract=True)

# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_extract=True)

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_extract=True)

# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_extract=True)



# ----- dynamic training start_dt_epoch='2' -----
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='2', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='2', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')





# ----- dynamic training start_dt_epoch='3' -----
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

                   

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True) 
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True) 
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True) 
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True) 


# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

                   
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', unkillable=True) 
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', unkillable=True) 
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', unkillable=True) 
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', unkillable=True) 
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', unkillable=True) 

# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

                   
                   
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='3', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', unkillable=True) 

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

                   
                   
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

                   
                   
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], unkillable=True) 


# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

                   
                   
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', data_model='', data_model_name=None,
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='3', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', unkillable=True) 





# ----- training on filtered datasets, based on training dynamics electra on deberta-----
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')



# from reboot
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

                   
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True)  

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)

quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)

quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='47',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True)

                   
                   
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)  
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6', from_reboot=True, unkillable=True)

# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True)

                   
                   
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='44',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='45',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='46',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26', from_reboot=True, unkillable=True) 

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)



quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 


# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True)


quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'], from_reboot=True, unkillable=True) 


# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True)




quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='44',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='45',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='46',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48', from_reboot=True, unkillable=True) 



                   
                   
# ----- training on filtered datasets, based on training dynamics electra on electra-----
# --- anli_v1.0_R3 ---
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

# --- boolq ---
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')
quick_write_config(workspace='huji', task='boolq', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='256', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['boolq'], burn_in='0',
                   burn_out='6')

# --- SNLI ---
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='42',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')
quick_write_config(workspace='huji', task='SNLI', train_set_fraction='0.1', model='google/electra-large-discriminator', model_type='electra',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='201', seed='43',
                   start_dt_epoch='0', patience='100', batch_size='2', eval_tasks_names=['SNLI', 'anli_v1.0_R3'],
                   num_eval_cycles='26', num_data_eval_cycles='26')

# --- WINOGRANDE ---
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- hellaswag ---
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])
quick_write_config(workspace='huji', task='hellaswag', batch_size='2', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='100', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'])

# --- abductive_nli ---
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.50', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.33', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='2', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='3', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='42',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')
quick_write_config(workspace='huji', task='abductive_nli', train_set_fraction='1', model='google/electra-large-discriminator', model_type='electra_mc',
                   model_name='electra-large', data_model='google/electra-large-discriminator', data_model_name='electra-large',
                   bias='4', metric='variability', favored_fraction='0.25', max_seq_length='null', seed='43',
                   start_dt_epoch='0', patience='100', eval_tasks_names=['hellaswag', 'abductive_nli', 'WINOGRANDE'],
                   num_eval_cycles='48', num_data_eval_cycles='48')

