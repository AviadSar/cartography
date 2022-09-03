import os

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
    "num_eval_cycles": {16},\n\
    "granularity": {17},\n\
    "metric": "{18}",\n\
    "bias": {19},\n\
    "favored_fraction": {20},\n\
    "start_dt_epoch": {21},\n\
    "td_dir": "{22}",\n\
    "overwrite_output_dir": true\n\
    "model_weights_output_dir": "{23}",\n\
    "save_model": {24}\n\
}}'

slurm_config_string = \
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

run_string = 'sbatch {}\n'


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
                 patience,
                 mem,
                 gres,
                 time,
                 train_set_fraction,
                 max_seq_length,
                 eval_steps,
                 num_eval_cycles,
                 num_data_eval_cycles,
                 granularity,
                 metric,
                 bias,
                 favored_fraction,
                 start_dt_epoch,
                 save_model,
                 ):
    config_dir = os.path.join('/cs/labs/roys/aviadsa/cartography/configs', task, model_name)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    if favored_fraction is None:
        td_dir = os.path.join('outputs', task, data_model_name,
                              '{}_{}_{}_128_batch_{}_evals'.format(workspace, task, data_model_name, num_data_eval_cycles))
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
            num_eval_cycles,
            granularity if granularity is not None else 'null',
            metric if metric is not None else 'null',
            bias if bias is not None else 'null',
            favored_fraction if favored_fraction is not None else 'null',
            start_dt_epoch if start_dt_epoch is not None else 'null',
            td_dir,
            model_weights_output_dir,
            save_model
        ))

    slurm_dir = os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_configs', task, model_name)
    if not os.path.exists(slurm_dir):
        os.makedirs(slurm_dir)

    slurm_out_dir = os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_out_files', task, model_name)
    if not os.path.exists(slurm_out_dir):
        os.makedirs(slurm_out_dir)

    with open(os.path.join(slurm_dir, file_name + '.sh'), 'w') as slurm_file:
        slurm_file.write(slurm_config_string.format(
            mem,
            gres,
            time,
            os.path.join(task, model_name, file_name),
            os.path.join(task, model_name, file_name)
        ))
        if data_model_name_or_path == '' and favored_fraction is None:
            slurm_file.write(filter_string.format(
                task,
                model_name,
                os.path.join(task, model_name, file_name),
                task
            ))

    consecutive = False
    if consecutive:
        with open(os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_configs', 'run_consecutive.sh'),
                  'a') as run_file:
            run_file.write(slurm_restricted_string.format(
                os.path.join(task, model_name, file_name),
                os.path.join(task, model_name, file_name)
            ))
            if data_model_name_or_path == '' and favored_fraction is None:
                run_file.write(filter_string.format(
                    task,
                    model_name,
                    os.path.join(task, model_name, file_name),
                    task
                ))

    with open(os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_configs/run_slurm.sh'), 'a') as run_file:
        run_file.write(run_string.format(os.path.join(slurm_dir, file_name + '.sh')))


def quick_write_config(workspace, task, model, model_type, data_dir_suffix='', cache_dir_suffix='', filtering='',
                       data_model='', model_name=None, data_model_name=None, seed='42', batch_size='4', num_epochs='5',
                       num_data_epochs='5', mem='16g', gres='gpu:rtx2080,vmem:10g', time='time=12:0:0',
                       train_set_fraction='1.0', max_seq_length='128', bias=None, gradient_accumulation='128',
                       eval_samples='12800', num_eval_cycles='24', num_data_eval_cycles='12', granularity=None,
                       metric=None, favored_fraction=None, start_dt_epoch=None, patience='100', save_model='true'):
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

    file_name = '{}_{}_{}{}{}{}{}{}{}{}_{}_batch_{}_evals_seed_{}'.format(workspace, task, model_name,
                                                '_on_' + data_model_name if data_model_name != '' else '',
                                                '_' + filtering if filtering != '' else '',
                                                '_bias_' + bias if bias is not None else '',
                                                '_' + metric if metric is not None else '',
                                                '_' + favored_fraction + '_dt_' if favored_fraction is not None else '',
                                                'strat_dt_epoch_' + start_dt_epoch + '_' if start_dt_epoch is not None else '',
                                                '_gran_' + granularity if granularity is not None else '',
                                                gradient_accumulation,
                                                num_eval_cycles,
                                                seed)
    if filtering != '':
        data_dir = os.path.join(data_dir_prefix, 'filtered_datasets',
                                'cartography_' + filtering + ('_bias_' + bias if bias is not None else ''), task,
                                '{}_{}_{}_{}_evals'.format(workspace, task, data_model_name, num_data_eval_cycles))
    else:
        data_dir = os.path.join(data_dir_prefix, task)
    cache_dir = os.path.join(cache_dir_prefix, task, 'cache_{}{}{}{}{}{}{}_batch_{}_{}'.format(model_name,
                                                                              '_on_' + data_model_name if data_model_name != '' else '',
                                                                              '_' + filtering if filtering != '' else '',
                                                                              '_bias_' + bias if bias is not None else '',
                                                                              '_' + metric if metric is not None else '',
                                                                              '_' + favored_fraction + '_dt_' if favored_fraction is not None else '',
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
                 patience=patience,
                 mem=mem,
                 gres=gres,
                 time=time,
                 train_set_fraction=train_set_fraction,
                 max_seq_length=max_seq_length,
                 eval_steps='{} / {}'.format(eval_samples, gradient_accumulation),
                 num_eval_cycles=num_eval_cycles,
                 num_data_eval_cycles=num_data_eval_cycles,
                 granularity=granularity,
                 metric=metric,
                 bias=bias,
                 favored_fraction=favored_fraction,
                 start_dt_epoch=start_dt_epoch,
                 save_model=save_model)


quick_write_config(workspace='huji', task='hellaswag', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='100', seed='42', batch_size='2')
quick_write_config(workspace='huji', task='hellaswag', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='80', seed='43', batch_size='2')
quick_write_config(workspace='huji', task='hellaswag', model='microsoft/deberta-large', model_type='deberta_mc',
                   model_name='deberta-large', max_seq_length='110', seed='44', batch_size='2')
quick_write_config(workspace='huji', task='boolq', model='microsoft/deberta-large', model_type='deberta',
                   model_name='deberta-large', max_seq_length='256', seed='42', batch_size='2')


# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', max_seq_length='null', seed='42')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', max_seq_length='null', seed='43')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', max_seq_length='null', seed='44')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', max_seq_length='null', seed='45')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', max_seq_length='null', seed='46')
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='1.2', metric='confidence', favored_fraction='0',
#                    max_seq_length='null')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='1.2', metric='variability', favored_fraction='0',
#                    max_seq_length='null')
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='42', start_dt_epoch='3')
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='43', start_dt_epoch='3')
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='44', start_dt_epoch='3')
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='45', start_dt_epoch='3')
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='1')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='2', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='confidence', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc',
#                    model_name='deberta-large', bias='3', metric='variability', favored_fraction='0.50',
#                    max_seq_length='null', seed='46', start_dt_epoch='3')




# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', model_type='electra', filtering='',
#                    model_name='electra-large', train_set_fraction='0.1', seed='43')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', model_type='deberta', filtering='',
#                    model_name='deberta-large', train_set_fraction='0.1')
# quick_write_config(workspace='huji', task='SNLI', model='roberta-large', model_type='roberta', filtering='',
#                    model_name='roberta-large', train_set_fraction='0.1')

# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', model_type='deberta_mc', filtering='',
#                    model_name='deberta-large', train_set_fraction='1.0', num_epochs='5', max_seq_length='null', batch_size='4')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', model_type='electra_mc', filtering='',
#                    model_name='electra-large', train_set_fraction='1.0', num_epochs='5', max_seq_length='null', batch_size='4', seed='43')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='roberta-large', model_type='roberta_mc', filtering='',
#                    model_name='roberta-large', train_set_fraction='1.0', num_epochs='5', max_seq_length='null', batch_size='4', seed='43')

# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='microsoft/deberta-large', model_type='deberta', filtering='',
#                    model_name='deberta-large', num_epochs='5', max_seq_length='null', batch_size='2', time='time=10:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='google/electra-large-discriminator', model_type='electra', filtering='',
#                    model_name='electra-large', num_epochs='5', max_seq_length='null', batch_size='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='roberta-large', model_type='roberta', filtering='',
#                    model_name='roberta-large', num_epochs='5', max_seq_length='null', batch_size='2')

# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta', filtering='',
#                    model_name='deberta-large', train_set_fraction='0.5', num_epochs='5', max_seq_length='201', batch_size='2', time='time=10:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra', filtering='',
#                    model_name='electra-large', train_set_fraction='0.5', num_epochs='5', max_seq_length='201', batch_size='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='roberta-large', model_type='roberta', filtering='',
#                    model_name='roberta-large', train_set_fraction='0.5', num_epochs='5', max_seq_length='201', batch_size='2')

# quick_write_config(workspace='huji', task='abductive_nli', model='microsoft/deberta-large', model_type='deberta_mc', filtering='',
#                    model_name='deberta-large', train_set_fraction='0.5', num_epochs='5', max_seq_length='null', batch_size='4')
# quick_write_config(workspace='huji', task='abductive_nli', model='google/electra-large-discriminator', model_type='electra_mc', filtering='',
#                    model_name='electra-large', train_set_fraction='0.5', num_epochs='5', max_seq_length='null', batch_size='4', time='time=10:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='roberta-large', model_type='roberta_mc', filtering='',
#                    model_name='roberta-large', train_set_fraction='0.5', num_epochs='5', max_seq_length='null', batch_size='4')


# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='google/electra-large-discriminator',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='electra-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta', filtering='random_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='SNLI', model='roberta-large', data_model='microsoft/deberta-large',
#                    model_type='roberta', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='deberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='SNLI', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='SNLI', model='roberta-large', data_model='google/electra-large-discriminator',
#                    model_type='roberta', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='electra-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='SNLI', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta', filtering='random_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='microsoft/deberta-large',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='deberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra', filtering='random_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', time='time=4:0:0')

# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='microsoft/deberta-large', data_model='google/electra-large-discriminator',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='electra-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta', filtering='random_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='roberta-large', data_model='microsoft/deberta-large',
#                    model_type='roberta', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='deberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='roberta-large', data_model='google/electra-large-discriminator',
#                    model_type='roberta', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='electra-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta', filtering='random_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='google/electra-large-discriminator', data_model='microsoft/deberta-large',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='deberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R1', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra', filtering='random_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
#
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='google/electra-large-discriminator',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='electra-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta', filtering='random_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='roberta-large', data_model='microsoft/deberta-large',
#                    model_type='roberta', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='deberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='roberta-large', data_model='google/electra-large-discriminator',
#                    model_type='roberta', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='electra-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta', filtering='random_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='microsoft/deberta-large',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='deberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', time='time=4:0:0')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra', filtering='random_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', time='time=4:0:0')
#
#
# quick_write_config(workspace='huji', task='abductive_nli', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='microsoft/deberta-large', data_model='google/electra-large-discriminator',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='electra-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta_mc', filtering='random_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='abductive_nli', model='roberta-large', data_model='microsoft/deberta-large',
#                    model_type='roberta_mc', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta_mc', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='roberta-large', data_model='google/electra-large-discriminator',
#                    model_type='roberta_mc', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='electra-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta_mc', filtering='random_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='abductive_nli', model='google/electra-large-discriminator', data_model='microsoft/deberta-large',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='deberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='abductive_nli', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra_mc', filtering='random_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')

# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='microsoft/deberta-large',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='deberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra_mc', filtering='random_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='google/electra-large-discriminator',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='electra-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta_mc', filtering='random_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='roberta-large', data_model='microsoft/deberta-large',
#                    model_type='roberta_mc', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta_mc', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='roberta-large', data_model='google/electra-large-discriminator',
#                    model_type='roberta_mc', filtering='variability_0.50', model_name='roberta-large',
#                    data_model_name='electra-large', max_seq_length='null', time='time=4:0:0')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='roberta-large', data_model='roberta-large',
#                    model_type='roberta_mc', filtering='random_0.50', model_name='roberta-large',
#                    data_model_name='roberta-large', max_seq_length='null', time='time=4:0:0')


# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra_mc', filtering='random_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', max_seq_length='null')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta_mc', filtering='random_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', max_seq_length='null')

# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra', filtering='random_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', max_seq_length='null')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta', filtering='random_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', max_seq_length='null')

# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='roberta-large',
#                    model_type='electra', filtering='random_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', max_seq_length='201', batch_size='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='roberta-large',
#                    model_type='deberta', filtering='random_0.50', model_name='deberta-large',
#                    data_model_name='roberta-large', max_seq_length='201', batch_size='2')

























# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.25', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.25', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.33', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.33', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='2')

# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.25', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.25', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.33', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.33', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='3')

# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.25', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.25', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.33', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.33', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra_mc', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta_mc', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='4')


# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.25', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.25', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.33', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.33', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='2')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='2')

# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.25', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.25', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.33', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.33', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='3')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='3')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.25', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.25', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.33', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.33', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='null', bias='4')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='null', bias='4')


# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.25', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2', bias='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.25', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2', bias='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.33', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2', bias='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.33', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2', bias='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2', bias='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2', bias='2')

# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.25', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2', bias='3')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.25', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2', bias='3')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.33', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2', bias='3')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.33', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2', bias='3')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2', bias='3')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2', bias='3')

# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.25', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2', bias='4')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.25', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2', bias='4')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.33', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2', bias='4')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.33', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2', bias='4')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator',
#                    model_type='electra', filtering='variability_0.50', model_name='electra-large',
#                    data_model_name='electra-large', max_seq_length='201', batch_size='2', bias='4')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', data_model='microsoft/deberta-large',
#                    model_type='deberta', filtering='variability_0.50', model_name='deberta-large',
#                    data_model_name='deberta-large', max_seq_length='201', batch_size='2', bias='4')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='2')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='2', train_set_fraction='0.1')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='2', train_set_fraction='0.1')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', gradient_accumulation='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', gradient_accumulation='2')
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='4')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='4')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='4', train_set_fraction='0.1')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='4', train_set_fraction='0.1')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', gradient_accumulation='4')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', gradient_accumulation='4')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='8')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='8')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='8', train_set_fraction='0.1')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='8', train_set_fraction='0.1')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', gradient_accumulation='8')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', gradient_accumulation='8')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='16')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='16')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='16', train_set_fraction='0.1')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='16', train_set_fraction='0.1')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', gradient_accumulation='16')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', gradient_accumulation='16')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='32')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='32', train_set_fraction='0.1')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='32', train_set_fraction='0.1')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', gradient_accumulation='32')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='64')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='64', train_set_fraction='0.1')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='64', train_set_fraction='0.1')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', gradient_accumulation='64')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='128')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='128', train_set_fraction='0.1')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='128', train_set_fraction='0.1')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', gradient_accumulation='128')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='256')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='256', train_set_fraction='0.1')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', gradient_accumulation='256', train_set_fraction='0.1')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', gradient_accumulation='256')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='2')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='2')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='2')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='2')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='2')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='2')
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='4')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='4')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='4')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='4')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='4')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='4')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='8')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='8')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='8')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='8')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='8')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='8')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='16')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='16')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='16')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='16')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='16')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='16')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='32')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='32')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='32')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='64')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='64')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='64')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='128')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='128')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='128')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='256')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='256')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='1', metric='variability', gradient_accumulation='256')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='8')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='8')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='8')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='8')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='8')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='8')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='16')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='16')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='16')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='16')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='16')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='16')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='32')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='32')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='32')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='64')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='64')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='64')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='128')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='128')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='128')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='256')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='256')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='4', metric='variability', gradient_accumulation='256')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='32')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='32')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='32')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='32')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='64')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='64')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='64')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='128')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='128')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='128')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='256')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='256')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='16', metric='variability', gradient_accumulation='256')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='64')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='64')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='64')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='64')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='128')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='128')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='128')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='128')
#
#
#
# quick_write_config(workspace='huji', task='WINOGRANDE', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra_mc', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='WINOGRANDE', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta_mc', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='256')
#
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large', max_seq_length='null',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='256')
#
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='google/electra-large-discriminator', data_model='google/electra-large-discriminator', data_model_name='electra-large',
#                    model_type='electra', model_name='electra-large',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='256')
# quick_write_config(workspace='huji', task='anli_v1.0_R3', max_seq_length='201', model='microsoft/deberta-large', data_model='microsoft/deberta-large', data_model_name='deberta-large',
#                    model_type='deberta', model_name='deberta-large',
#                    batch_size='2', granularity='32', metric='variability', gradient_accumulation='256')