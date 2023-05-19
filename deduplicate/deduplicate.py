import datasets
import soundfile
import numpy as np
from datasets import load_dataset
from utils import Deduplicate

data_link = "quocanh34/wer0_cut_v1"

# load dataset and add sum column
ds = datasets.load_dataset(data_link)


def add_sum(example):
    example["sum"] = np.sum(example["audio"]["array"])
    return example


updated_ds = ds["train"].map(add_sum, num_proc=8)

# deduplicate
deduplicate = Deduplicate(updated_ds)
deduplicated_ds = deduplicate.run_deduplicate()

deduplicated_ds.push_to_hub('linhtran92/deduplicated_dataset_100hrs_wer0_cut',
                            token='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')
