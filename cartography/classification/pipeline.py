import os

os.system("python run_glue.py -c ../../configs/snli_roberta.jsonnet --do_train -o ../../outputs/try_roberta/")
os.system("python run_glue.py -c ../../configs/snli_bert.jsonnet --do_train -o ../../outputs/try_bert/")

os.system("python ../selection/train_dy_filtering.py --plot --task_name SNLI --metric variability --model_dir ../../outputs/try_roberta --data_dir C:/home/swabhas/data/glue/")
os.system("python ../selection/train_dy_filtering.py --plot --task_name SNLI --metric variability --model_dir ../../outputs/try_bert --data_dir C:/home/swabhas/data/glue/")


os.system("python ../selection/train_dy_filtering.py --filter --task_name SNLI --metric variability --model_dir ../../outputs/try_roberta --data_dir C:/home/swabhas/data/glue/")
os.system("python ../selection/train_dy_filtering.py --filter --task_name SNLI --metric variability --model_dir ../../outputs/try_bert --data_dir C:/home/swabhas/data/glue/")