import datasets
from datasets import load_dataset, Dataset, concatenate_datasets

ds = load_dataset("quocanh34/youtube_dataset_new_cut_final")
ds = ds.shuffle(seed=42)

def split_dataset(dataset, order):
    index_list = list(range(dataset['train'].num_rows))
    split_index = int(len(index_list)/3)
    if order == 1: 
        return dataset['train'].select(index_list[:split_index]).train_test_split(test_size=0.05)
    if order == 2: 
        return dataset['train'].select(index_list[split_index : 2*split_index]).train_test_split(test_size=0.05)
    if order == 3: 
        return dataset['train'].select(index_list[2*split_index:]).train_test_split(test_size=0.05)

ds_1 = split_dataset(ds, 1)
ds_2 = split_dataset(ds, 2)
ds_3 = split_dataset(ds, 3)

ds_1.push_to_hub("quocanh34/viet_youtube_asr_v1", token="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")
ds_2.push_to_hub("quocanh34/viet_youtube_asr_v2", token="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")
ds_3.push_to_hub("quocanh34/viet_youtube_asr_v3", token="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")

print(f"Total rows: {ds['train'].num_rows}")
print(f"ds1: {ds_1}")
print(f"ds2: {ds_2}")
print(f"ds3: {ds_3}")




