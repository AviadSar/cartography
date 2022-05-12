import os

# os.system("python run_glue.py -c ../../configs/snli_roberta.jsonnet --do_train -o ../../outputs/snli_5_roberta_whole_set/")
# os.system("python run_glue.py -c ../../configs/snli_bert.jsonnet --do_train -o ../../outputs/snli_10_bert/")


# os.system("python run_glue.py -c ../../configs/snli_roberta_on_random_033.jsonnet --do_train -o ../../outputs/snli_roberta_on_random_033/")
# os.system("python run_glue.py -c ../../configs/snli_bert_on_random_033.jsonnet --do_train -o ../../outputs/snli_bert_on_random_033/")


os.system("python ../selection/train_dy_filtering.py --plot --task_name SNLI --metric variability --model roberta-base --model_dir ../../outputs/snli_5_roberta_whole_set --data_dir C:/my_documents/datasets/swabhas/data/glue/SNLI --plots_dir ../../cartography/plots --filtering_output_dir ../../cartography/filtered_datasets")
# os.system("python ../selection/train_dy_filtering.py --plot --task_name SNLI --metric variability --model bert-base --model_dir ../../outputs/snli_10_bert --data_dir C:/home/swabhas/data/glue/SNLI")
# os.system("python ../selection/train_dy_filtering.py --filter --task_name SNLI --metric variability --model roberta-base --model_dir ../../outputs/snli_5_roberta_whole_set --data_dir C:/my_documents/datasets/swabhas/data/glue/SNLI")
# os.system("python ../selection/train_dy_filtering.py --filter --task_name SNLI --metric variability --model bert-base --model_dir ../../outputs/snli_10_bert --data_dir C:/home/swabhas/data/glue/SNLI")