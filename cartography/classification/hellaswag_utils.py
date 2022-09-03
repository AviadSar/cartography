import os
import json

from transformers.data.processors.utils import DataProcessor, InputExample

from cartography.classification.multiple_choice_utils import MCInputExample
from cartography.data_utils import read_data


class HellaSwagProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def get_labels(self):
        return [0, 1, 2, 3]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = f"{set_type}-{i}" #"%s-%s" % (set_type, line[0])
            guid = i

            context = line['ctx']

            format_str = '{} {}'
            option0 = format_str.format(context, line['endings'][0])
            option1 = format_str.format(context, line['endings'][1])
            option2 = format_str.format(context, line['endings'][2])
            option3 = format_str.format(context, line['endings'][3])

            label = line['label']

            mc_example = MCInputExample(
                example_id=int(guid),
                contexts=[context, context, context, context],
                question='',
                endings=[option0, option1, option2, option3],
                label=label
            )

            if label not in self.get_labels():
                continue

            examples.append(mc_example)
        return examples

    def get_examples(self, data_file, set_type):
        with open(data_file, 'r') as json_file:
            return self._create_examples([json.loads(line) for line in json_file], set_type=set_type)
        raise IOError('could not open file: {}'.format(data_file))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "test.jsonl"), "test")