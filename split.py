import datasets
from datasets import Dataset, load_dataset, DatasetDict

def get_split(dataset, split, order, test_size):
    if order < 1 or order > split:
        raise ValueError('error')
    split_len = len(dataset['train']) // split   
    start = (order - 1) * split_len
    end = start + split_len if order != split else len(dataset['train']) 
    ds_split_train_test = dataset['train'].select(range(start, end)).train_test_split(test_size)
    train_ds, test_ds = ds_split_train_test["train"], ds_split_train_test["test"]
    ds_split_train_val = train_ds.train_test_split(test_size/(1-test_size))
    train_ds, val_ds = ds_split_train_val["train"], ds_split_train_val["test"]

    return DatasetDict({"train": train_ds,
                        "test": test_ds,
                        "validation": val_ds})

ds = load_dataset("linhtran92/dataset_500hrs_wer0")
ds = ds.shuffle(seed=42)

ds_1 = get_split(ds, split=3, order=1, test_size=7500/649158)
ds_2 = get_split(ds, split=2, order=2, test_size=7500/649158)

print(f"Total rows: {ds['train'].num_rows}")
print(f"ds1: {ds_1}")
print(f"ds2: {ds_2}")


ds_1.push_to_hub("quocanh34/viet_asr_corpus_v1", token="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")
ds_2.push_to_hub("quocanh34/viet_asr_corpus_v2", token='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')

print(f"ds: {ds}")
print(f"ds1: {ds_1}")
print(f"ds2: {ds_2}")



