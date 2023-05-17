import datasets
import pandas as pd
import numpy as np
from datasets import load_dataset, concatenate_datasets
from collections import Counter

updated_ds = ""


def create_dict(dataset):
    freq_sum = dataset['sum']
    idx = list(range(len(freq_sum)))
    return dict(zip(idx, freq_sum))


my_dict = create_dict(updated_ds)


def duplicated_index(my_dict):
    counts = Counter(my_dict.values())
    cut_idx = [k for k, v in my_dict.items() if counts[v] > 1]
    keep_idx = [k for k, v in my_dict.items() if counts[v] == 1]
    return cut_idx, keep_idx


cut_idx, keep_idx = duplicated_index(my_dict)


def create_potential_duplicate(dataset, cut_idx, keep_idx):
    dup_ds = dataset.select(cut_idx)
    undup_ds = dataset.select(keep_idx)
    return dup_ds, undup_ds


dup_ds, undup_ds = create_potential_duplicate(updated_ds, cut_idx, keep_idx)


def cut_duplicated_sample(dup_ds):
    df = pd.DataFrame(dup_ds)
    check = list(df.duplicated(subset=['w2v2_transcription', 'sum']))
    print("created check list...")
    indices = [index for (index, item) in enumerate(check) if item == False]
    print("get indices finished...")
    updated_undup_ds = dup_ds.select(indices)
    return updated_undup_ds


def run_deduplicate(update_ds, cut_idx, keep_idx, dup_ds, undup_ds):
    print(f"Before deduplicate : {updated_ds.num_rows} samples")
    print(f"There are {len(cut_idx)} duplicated in audio array sum...")
    print("Split dataset into duplicate and unduplicate complete...")
    print("Duplicated ds\n", dup_ds, "Keep ds\n", undup_ds)
    updated_undup_ds = cut_duplicated_sample(dup_ds)
    print("deduplicate finished...")
    print(updated_undup_ds)
    deduplicated_ds = concatenate_datasets([undup_ds, updated_undup_ds])
    print(deduplicated_ds)
    print(f"After deduplicate : {deduplicated_ds.num_rows} samples")
    return deduplicated_ds


deduplicated_ds = run_deduplicate(
    updated_ds, cut_idx, keep_idx, dup_ds, undup_ds)
