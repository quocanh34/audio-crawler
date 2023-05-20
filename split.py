import datasets
from datasets import Dataset, load_dataset

def get_split(dataset, split, order, test_size):
    if order < 1 or order > split:
        raise ValueError('error')
    
    split_len = len(dataset['train']) // split   
    start = (order - 1) * split_len
    end = start + split_len if order != split else len(dataset['train']) 
    return dataset['train'].select(range(start, end)).train_test_split(test_size)

ds = load_dataset("linhtran92/final_dataset_500hrs_wer0")
ds = ds.shuffle(seed=42)

ds_1 = get_split(ds, split=3, order=1, test_size=0.1)
ds_2 = get_split(ds, split=3, order=2, test_size=0.1)
ds_3 = get_split(ds, split=3, order=3, test_size=0.1)

print(f"Total rows: {ds['train'].num_rows}")
print(f"ds1: {ds_1}")
print(f"ds2: {ds_2}")
print(f"ds3: {ds_3}")

ds_1.push_to_hub("quocanh34/viet_youtube_asr_corpus_v1", token="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")
ds_2.push_to_hub("linhtran92/viet_youtube_asr_corpus_v2", token='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')
ds_3.push_to_hub("quocanh34/viet_youtube_asr_corpus_v3", token="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")




