local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_VISIBLE_DEVICES"));

local LEARNING_RATE = 1.0708609960508476e-05;
local BATCH_SIZE = 4;
local NUM_EPOCHS = 5;
local SEED = 93078;

local TASK = "SNLI";
local DATA_DIR = "/cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets/cartography_variability_0.33/SNLI/huji_snli_5_bert";
local FEATURES_CACHE_DIR = "/cs/labs/roys/aviadsa/datasets/swabhas/data/glue/" + TASK + "/cache_roberta_on_bert_033_" + SEED;

local TEST = "C:\\my_documents\\datasets\\swabhas\\data\\glue\\SNLI\\diagnostic_test.tsv";

{
   "data_dir": DATA_DIR,
   "model_type": "bert",
   "model_name_or_path": "roberta-base",
   "data_model_name_or_path": "bert-base-uncased",
   "task_name": TASK,
   "seed": SEED,
   "num_train_epochs": NUM_EPOCHS,
   "learning_rate": LEARNING_RATE,
   "features_cache_dir": FEATURES_CACHE_DIR,
   "per_gpu_train_batch_size": BATCH_SIZE,
   "per_gpu_eval_batch_size": BATCH_SIZE,
   "gradient_accumulation_steps": 128 / BATCH_SIZE,
   "do_train": true,
   "do_eval": true,
   "do_test": true,
   "test": TEST,
   "patience": 5,
}