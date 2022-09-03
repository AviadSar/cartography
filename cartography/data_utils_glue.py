import logging
import random
import re
import tqdm
import json
from collections import OrderedDict

logger = logging.getLogger(__name__)


def convert_string_to_unique_number(string: str) -> int:
    """
    Hack to convert SNLI ID into a unique integer ID, for tensorizing.
    """
    id_map = {'e': '0', 'c': '1', 'n': '2'}

    # SNLI-specific hacks.
    if string.startswith('vg_len'):
        code = '555'
    elif string.startswith('vg_verb'):
        code = '444'
    else:
        code = '000'

    try:
        number = int(code + re.sub(r"\D", "", string) + id_map.get(string[-1], '3'))
    except:
        number = random.randint(10000, 99999)
        logger.info(f"Cannot find ID for {string}, using random number {number}.")
    return number


def read_jsonl_task(file_path: str,
                    guid_index: int = None,
                    label: str = 'label',
                    labels=[]):
    with open(file_path, 'r') as jsonl_file:
        i = -1
        jsonl_dict = OrderedDict
        for line in tqdm.tqdm([line for line in jsonl_file]):
            i += 1
            json_line = json.loads(line)
            if not json_line[label] in labels:
                i -= 1
                continue
            jsonl_dict[i] = line
        return jsonl_dict, None


def read_glue_tsv(file_path: str,
                  guid_index: int,
                  label_index: int = -1,
                  guid_as_int: bool = False):
    """
    Reads TSV files for GLUE-style text classification tasks.
    Returns:
      - a mapping between the example ID and the entire line as a string.
      - the header of the TSV file.
    """
    tsv_dict = OrderedDict()

    i = -1
    with open(file_path, 'r') as tsv_file:
        for line in tqdm.tqdm([line for line in tsv_file]):
            i += 1
            if i == 0:
                header = line.strip()
                field_names = line.strip().split("\t")
                continue

            fields = line.strip().split("\t")
            label = fields[label_index]

            # TODO: SUPER HACKY AND BUGGY CODE FROM OLD CARTOGRAPHY REPO
            # if len(fields) > len(field_names):
            #     # SNLI / MNLI fields sometimes contain multiple annotator labels.
            #     # Ignore all except the gold label.
            #     reformatted_fields = fields[:len(field_names) - 1] + [label]
            #     assert len(reformatted_fields) == len(field_names)
            #     reformatted_line = "\t".join(reformatted_fields)
            # else:
            #     reformatted_line = line.strip()
            reformatted_line = line.strip()

            if label == "-" or label == "":
                logger.info(f"Skippping line: {line}")
                i -= 1
                continue

            if guid_index is None:
                guid = i - 1
            else:
                guid = fields[guid_index]  # PairID.
            # TODO: resolve no skip. basically we want to allow ids to be identicall in our filtered sets,
            #  but not in original datasets
            # if guid in tsv_dict:
            #     logger.info(f"Found clash in IDs ... skipping example {guid}.")
            #     continue
            tsv_dict[guid] = reformatted_line.strip()

    logger.info(f"Read {len(tsv_dict)} valid examples, with unique IDS, out of {i} from {file_path}")
    if guid_as_int:
        tsv_numeric = {int(convert_string_to_unique_number(k)): v for k, v in tsv_dict.items()}
        return tsv_numeric, header
    return tsv_dict, header
