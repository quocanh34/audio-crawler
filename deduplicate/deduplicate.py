import datasets
import soundfile
import numpy as np
from datasets import load_dataset
from utils import Deduplicate

data_link = "quocanh34/youtube_dataset_locfuho"

# load dataset and add sum column
ds = datasets.load_dataset(data_link)


def add_sum(example):
    example["sum"] = np.sum(example["audio"]["array"])
    return example


updated_ds = ds["train"].map(add_sum)

# deduplicate
deduplicate = Deduplicate(updated_ds)
deduplicated_ds = deduplicate.run_deduplicate()
