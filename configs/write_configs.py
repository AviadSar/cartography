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
local FEATURES_CACHE_DIR = {6};\n\
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
}}'

slurm_config_string = \
'#!/bin/bash\n\
#SBATCH --mem={0}\n\
#SBATCH --gres={1}\n\
#SBATCH --{2}\n\
#SBATCH --output=/cs/labs/roys/aviadsa/cartography/slurm_out_files/{3}.txt\n\
\n\
python cartography/classification/run_glue.py -c configs/{3}.jsonnet --do_train -o outputs/{4}/\n'


def write_config(file_name,
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
                 mem,
                 gres,
                 time):
    with open(os.path.join('/cs/labs/roys/aviadsa/cartography/configs', file_name + '.jsonnet'), 'w') as config_file:
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
            gradient_accumulation_steps))

    with open(os.path.join('/cs/labs/roys/aviadsa/cartography/slurm_configs', file_name + '.sh'), 'w') as slurm_file:
        slurm_file.write(slurm_config_string.format(
            mem,
            gres,
            time,
            file_name,
            file_name + '_' + num_epochs + '_epochs'
        ))


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

write_config(file_name='huji_winogrande_bert_large_on_bert_large_033',
             learning_rate='1.1235456034244052e-05',
             batch_size='4',
             num_epochs='5',
             seed='71789',
             task='WINOGRANDE',
             data_dir='/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.33/SNLI/huji_winogrande_bert_large_5_epochs',
             cache_dir='"/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/cache_bert_large_on_bert_large_033_" + SEED',
             test_dir='/cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV/diagnostic_test.tsv',
             model_type='bert_mc',
             model_name_or_path='bert-large-uncased',
             data_model_name_or_path='bert-large-uncased',
             gradient_accumulation_steps='128 / BATCH_SIZE',
             mem='8g',
             gres='gpu:rtx2080,vmem:10g',
             time='time=3:0:0')
