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
    "patience": 5,\n\
    "train_set_fraction": {12},\n\
    "max_seq_length": {13},\n\
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


def write_config(file_name,
                 learning_rate,
                 batch_size,
                 num_epochs,
                 seed,
                 task,
                 data_dir,
                 cache_dir,
                 test_dir,
                 model_name,
                 model_type,
                 model_name_or_path,
                 data_model_name_or_path,
                 gradient_accumulation_steps,
                 mem,
                 gres,
                 time,
                 train_set_fraction,
                 max_seq_length):
    config_dir = os.path.join('/cs/labs/roys/aviadsa/cartography/configs', task, model_name)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

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
            train_set_fraction,
            max_seq_length))

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
            os.path.join(task, model_name, '{}_{}_epochs'.format(file_name, num_epochs))
        ))
        if data_model_name_or_path == '':
            slurm_file.write(filter_string.format(
                task,
                model_name,
                os.path.join(task, model_name, '{}_{}_epochs'.format(file_name, num_epochs)),
                task
            ))

    consecutive = False
    if consecutive:
        with open(os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_configs', 'run_consecutive.sh'),
                  'a') as run_file:
            run_file.write(slurm_restricted_string.format(
                os.path.join(task, model_name, file_name),
                os.path.join(task, model_name, '{}_{}_epochs'.format(file_name, num_epochs))
            ))
            if data_model_name_or_path == '':
                run_file.write(filter_string.format(
                    task,
                    model_name,
                    os.path.join(task, model_name, '{}_{}_epochs'.format(file_name, num_epochs)),
                    task
                ))

    with open(os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_configs/run_slurm.sh'), 'a') as run_file:
        run_file.write(run_string.format(os.path.join(slurm_dir, file_name + '.sh')))


def quick_write_config(workspace, task, model, model_type, data_dir_suffix='', cache_dir_suffix='', filtering='',
                       data_model='', model_name=None, data_model_name=None, seed='42', batch_size='4', num_epochs='5',
                       num_data_epochs='5', mem='8g', gres='gpu:rtx2080,vmem:10g', time='time=8:0:0',
                       train_set_fraction='1.0', max_seq_length='128'):
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

    file_name = '{}_{}_{}{}{}_{}_epochs'.format(workspace, task, model_name,
                                                '_on_' + data_model_name if data_model_name != '' else '',
                                                '_' + filtering if filtering != '' else '',
                                                num_epochs)

    if filtering != '':
        data_dir = os.path.join(data_dir_prefix, 'filtered_datasets', 'cartography_' + filtering, task,
                                '{}_{}_{}_{}_epochs'.format(workspace, task, data_model_name, num_data_epochs))
    else:
        data_dir = os.path.join(data_dir_prefix, task)
    cache_dir = os.path.join(cache_dir_prefix, task, 'cache_{}{}{}_{}'.format(model_name,
                                                                              '_on_' + data_model_name if data_model_name != '' else '',
                                                                              '_' + filtering if filtering != '' else '',
                                                                              seed))

    write_config(file_name=file_name,
                 learning_rate='1.0708609960508476e-05',
                 batch_size=batch_size,
                 num_epochs=num_epochs,
                 seed=seed,
                 task=task,
                 data_dir=data_dir,
                 cache_dir=cache_dir,
                 test_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE/diagnostic_test.tsv',
                 model_name=model_name,
                 model_type=model_type,
                 model_name_or_path=model,
                 data_model_name_or_path=data_model,
                 gradient_accumulation_steps='128 / BATCH_SIZE',
                 mem=mem,
                 gres=gres,
                 time=time,
                 train_set_fraction=train_set_fraction,
                 max_seq_length=max_seq_length)


# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator', model_type='electra', filtering='',
#                    model_name='electra-large', train_set_fraction=0.1)
# quick_write_config(workspace='huji', task='SNLI', model='microsoft/deberta-large', model_type='deberta', filtering='',
#                    model_name='deberta-large', train_set_fraction=0.1)
# quick_write_config(workspace='huji', task='SNLI', model='t5-large', model_type='t5', filtering='',
#                    model_name='t5-large', train_set_fraction=0.1, num_epochs='10')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='microsoft/deberta-large', model_type='deberta', filtering='',
                   model_name='deberta-large', train_set_fraction='1.0', num_epochs='5', max_seq_length='201', batch_size='2')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='google/electra-large-discriminator', model_type='electra', filtering='',
                   model_name='electra-large', train_set_fraction='1.0', num_epochs='5', max_seq_length='201', batch_size='2')
quick_write_config(workspace='huji', task='anli_v1.0_R3', model='roberta-large', model_type='roberta', filtering='',
                   model_name='roberta-large', train_set_fraction='1.0', num_epochs='5', max_seq_length='201', batch_size='2')

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
# quick_write_config(workspace='huji', task='SNLI', model='google/electra-large-discriminator',
#                    data_model='roberta-large',
#                    model_type='electra', filtering='random_0.50', model_name='electra-large',
#                    data_model_name='roberta-large', time='time=8:0:0', seed='43')

# write_config(file_name='huji_snli_bert_on_bert_033',
#              learning_rate='1.0708609960508476e-05',
#              batch_size='4',
#              num_epochs='5',
#              seed='93078',
#              task='SNLI',
#              data_dir='/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.33/SNLI/huji_snli_5_bert',
#              cache_dir='"/cs/labs/roys/aviadsa/datasets/swabhas/data/glue/" + TASK + "/cache_bert_on_bert_033_" + SEED',
#              test_dir='C:\\my_documents\\datasets\\swabhas\\data\\glue\\SNLI\\diagnostic_test.tsv',
#              model_type='bert',
#              model_name_or_path='bert-base-uncased',
#              data_model_name_or_path='bert-base-uncased',
#              gradient_accumulation_steps='128 / BATCH_SIZE',
#              mem='8g',
#              gres='gpu:rtx2080,vmem:10g',
#              time='time=3:0:0')

# write_config(file_name='huji_winogrande_bert_large',
#              learning_rate='1.1235456034244052e-05',
#              batch_size='4',
#              num_epochs='5',
#              seed='71789',
#              task='WINOGRANDE',
#              data_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV',
#              cache_dir='DATA_DIR + "/cache_roberta_" + SEED',
#              test_dir='/cs/labs/roys/aviadsa/datasets/swabhas/data/glue/SNLI/diagnostic_test.tsv',
#              model_type='bert_mc',
#              model_name_or_path='bert-large-uncased',
#              data_model_name_or_path='bert-large-uncased',
#              gradient_accumulation_steps='128 / BATCH_SIZE',
#              mem='8g',
#              gres='gpu:rtx2080,vmem:10g',
#              time='time=8:0:0')

# write_config(file_name='huji_winogrande_roberta_on_roberta_0.50',
#              learning_rate='1.0708609960508476e-05',
#              batch_size='4',
#              num_epochs='10',
#              seed='71789',
#              task='WINOGRANDE',
#              data_dir='/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.50/WINOGRANDE/huji_winogrande_roberta_large_5_epochs',
#              cache_dir='"/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/cache_roberta_on_roberta_0.50_" + SEED',
#              test_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/diagnostic_test.tsv',
#              model_type='roberta_mc',
#              model_name_or_path='roberta-large',
#              data_model_name_or_path='roberta-large',
#              gradient_accumulation_steps='128 / BATCH_SIZE',
#              mem='8g',
#              gres='gpu:rtx2080,vmem:10g',
#              time='time=4:0:0')
#
# write_config(file_name='huji_winogrande_roberta_on_deberta_0.50',
#              learning_rate='1.0708609960508476e-05',
#              batch_size='4',
#              num_epochs='10',
#              seed='71789',
#              task='WINOGRANDE',
#              data_dir='/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.50/WINOGRANDE/huji_winogrande_deberta_large_5_epochs',
#              cache_dir='"/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/cache_roberta_on_deberta_0.50_" + SEED',
#              test_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/diagnostic_test.tsv',
#              model_type='roberta_mc',
#              model_name_or_path='roberta-large',
#              data_model_name_or_path='microsoft/deberta-large',
#              gradient_accumulation_steps='128 / BATCH_SIZE',
#              mem='8g',
#              gres='gpu:rtx2080,vmem:10g',
#              time='time=4:0:0')
#
# write_config(file_name='huji_winogrande_roberta_on_random_0.50',
#              learning_rate='1.0708609960508476e-05',
#              batch_size='4',
#              num_epochs='10',
#              seed='71789',
#              task='WINOGRANDE',
#              data_dir='/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_random_0.50/WINOGRANDE/huji_winogrande_roberta_large_5_epochs',
#              cache_dir='"/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/cache_roberta_on_random_0.50_" + SEED',
#              test_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/diagnostic_test.tsv',
#              model_type='roberta_mc',
#              model_name_or_path='roberta-large',
#              data_model_name_or_path='roberta-large',
#              gradient_accumulation_steps='128 / BATCH_SIZE',
#              mem='8g',
#              gres='gpu:rtx2080,vmem:10g',
#              time='time=4:0:0')
#
# write_config(file_name='huji_winogrande_deberta_on_deberta_0.50',
#              learning_rate='1.0708609960508476e-05',
#              batch_size='2',
#              num_epochs='10',
#              seed='71789',
#              task='WINOGRANDE',
#              data_dir='/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.50/WINOGRANDE/huji_winogrande_deberta_large_5_epochs',
#              cache_dir='"/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/cache_deberta_on_deberta_0.50_" + SEED',
#              test_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/diagnostic_test.tsv',
#              model_type='deberta_mc',
#              model_name_or_path='microsoft/deberta-large',
#              data_model_name_or_path='microsoft/deberta-large',
#              gradient_accumulation_steps='128 / BATCH_SIZE',
#              mem='8g',
#              gres='gpu:rtx2080,vmem:10g',
#              time='time=4:0:0')
#
# write_config(file_name='huji_winogrande_deberta_on_roberta_0.50',
#              learning_rate='1.0708609960508476e-05',
#              batch_size='2',
#              num_epochs='10',
#              seed='71789',
#              task='WINOGRANDE',
#              data_dir='/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.50/WINOGRANDE/huji_winogrande_roberta_large_5_epochs',
#              cache_dir='"/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/cache_deberta_on_roberta_0.50_" + SEED',
#              test_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/diagnostic_test.tsv',
#              model_type='deberta_mc',
#              model_name_or_path='microsoft/deberta-large',
#              data_model_name_or_path='roberta-large',
#              gradient_accumulation_steps='128 / BATCH_SIZE',
#              mem='8g',
#              gres='gpu:rtx2080,vmem:10g',
#              time='time=4:0:0')
#
# write_config(file_name='huji_winogrande_deberta_on_random_0.50',
#              learning_rate='1.0708609960508476e-05',
#              batch_size='2',
#              num_epochs='10',
#              seed='71789',
#              task='WINOGRANDE',
#              data_dir='/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_random_0.50/WINOGRANDE/huji_winogrande_roberta_large_5_epochs',
#              cache_dir='"/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/cache_deberta_on_random_0.50_" + SEED',
#              test_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/diagnostic_test.tsv',
#              model_type='deberta_mc',
#              model_name_or_path='microsoft/deberta-large',
#              data_model_name_or_path='roberta-large',
#              gradient_accumulation_steps='128 / BATCH_SIZE',
#              mem='8g',
#              gres='gpu:rtx2080,vmem:10g',
#              time='time=4:0:0')
