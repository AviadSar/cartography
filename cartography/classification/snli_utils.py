import os

from transformers.data.processors.utils import DataProcessor, InputExample
from cartography.data_utils import read_data


class SNLIProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def get_labels(self):
        return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, records, set_type):
        """Creates examples for the training and dev sets."""
        tsv_dict, header = records
        examples = []
        for idx, line in tsv_dict.items():
            fields = line.strip().split("\t")
            text_a = fields[5]
            text_b = fields[6]
            label = fields[0]

            examples.append(
                InputExample(guid=idx, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_examples(self, data_file, set_type):
        return self._create_examples(read_data(data_file, task_name="SNLI"), set_type)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "train.tsv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "dev.tsv"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "test.tsv"), "test")
