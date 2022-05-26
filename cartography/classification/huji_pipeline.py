import os
import sys

if sys.path[0] != '.':
    print('first path variable is: ' + sys.path[0])
    sys.path.insert(0, '.')
    print("added '.' to sys.path")

# os.system("python cartography/classification/run_glue.py -c configs/snli_roberta.jsonnet --do_train -o outputs/snli_5_roberta_whole_set/")

# os.system("python cartography/classification/run_glue.py -c configs/snli_roberta_on_random_033.jsonnet --do_train -o outputs/snli_roberta_on_random_033/")

# os.system("python cartography/selection/train_dy_filtering.py --plot --task_name WINOGRANDE --metric variability --model roberta-base --model_dir outputs/huji_winogrande_5_roberta --data_dir /cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV")
# os.system("python cartography/selection/train_dy_filtering.py --filter --task_name WINOGRANDE --metric variability --model roberta-base --model_dir outputs/huji_winogrande_5_roberta --data_dir /cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV")

# os.system("python cartography/selection/train_dy_filtering.py --plot --filter --task_name SNLI --metric variability --model bert-base-uncased --model_dir outputs/huji_snli_5_bert --data_dir /cs/labs/roys/aviadsa/datasets/swabhas/data/glue/SNLI  --plots_dir /cs/labs/roys/aviadsa/cartography/cartography/plots --filtering_output_dir /cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets")
# os.system("python cartography/selection/train_dy_filtering.py --plot --filter --task_name WINOGRANDE --metric variability --model bert-large-uncased --model_dir outputs/huji_winogrande_bert_large3_5_epochs --data_dir /cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV  --plots_dir /cs/labs/roys/aviadsa/cartography/cartography/plots --filtering_output_dir /cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets")
os.system("python cartography/selection/train_dy_filtering.py --plot --filter --task_name WINOGRANDE --metric variability --model roberta-large-uncased --model_dir outputs/huji_winogrande_roberta_large_5_epochs --data_dir /cs/labs/roys/aviadsa/datasets/WINOGRANDE_TSV  --plots_dir /cs/labs/roys/aviadsa/cartography/cartography/plots --filtering_output_dir /cs/labs/roys/aviadsa/cartography/cartography/filtered_datasets")
