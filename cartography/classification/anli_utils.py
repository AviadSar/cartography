import os
import json

from transformers.data.processors.utils import DataProcessor, InputExample


class ANLIProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def get_labels(self):
        return ["e", "n", "c"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = f"{set_type}-{i}" #"%s-%s" % (set_type, line[0])
            guid = i
            text_a = line['context']
            text_b = line['hypothesis']
            label = line['label']

            if label != "n" and label != "e" and label != "c":
                continue

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_examples(self, data_file, set_type):
        with open(data_file, 'r') as json_file:
            return self._create_examples([json.loads(line) for line in json_file], set_type=set_type)
        raise IOError('could no open file: {}'.format(data_file))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "test.jsonl"), "test")
