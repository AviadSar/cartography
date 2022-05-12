config_string =\
'\
local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_VISIBLE_DEVICES"));\
\
local LEARNING_RATE = {0};\
local BATCH_SIZE = {1};\
local NUM_EPOCHS = {2}};\
local SEED = {3};\
\
local TASK = "{4}";\
local DATA_DIR = "{5}";\
local FEATURES_CACHE_DIR = {6};\
\
local TEST = "{7}";\
\
{\
   "data_dir": DATA_DIR,\
   "model_type": "{8}",\
   "model_name_or_path": "{9}",\
   "data_model_name_or_path": "{10}",\
   "task_name": TASK,\
   "seed": SEED,\
   "num_train_epochs": NUM_EPOCHS,\
   "learning_rate": LEARNING_RATE,\
   "features_cache_dir": FEATURES_CACHE_DIR,\
   "per_gpu_train_batch_size": BATCH_SIZE,\
   "per_gpu_eval_batch_size": BATCH_SIZE,\
   "gradient_accumulation_steps": {11},\
   "do_train": true,\
   "do_eval": true,\
   "do_test": true,\
   "test": TEST,\
   "patience": 5,\
}\
'

slurm_config_string =\
'\
#!/bin/bash\
#SBATCH --mem=8g\
#SBATCH --gres=gpu:rtx2080,vmem:10g\
#SBATCH --time=3:0:0\
#SBATCH --output=/cs/labs/roys/aviadsa/cartography/slurm_out_files/huji_snli_bert_on_bert_033.txt\
\
python cartography/classification/run_glue.py -c configs/huji_snli_bert_on_bert_033.jsonnet --do_train -o outputs/huji_snli_5_bert_on_bert_033/\
'

write_config(file_name='huji_snli_bert_on_bert_033',
             learning_rate='1.0708609960508476e-05',
             batch_size='4',
             num_epochs='5',
             seed='93078',
             task='SNLI',
             data_dir='/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.33/SNLI/huji_snli_5_bert',
             cache_dir='"/cs/labs/roys/aviadsa/datasets/swabhas/data/glue/" + TASK + "/cache_bert_on_bert_033_" + SEED',
             test_dir='C:\\my_documents\\datasets\\swabhas\\data\\glue\\SNLI\\diagnostic_test.tsv',
             model_type='bert',
             model_name_or_path='bert-base-uncased',
             data_model_name_or_path='bert-base-uncased',
             gradient_accumulation_steps='128 / BATCH_SIZE')