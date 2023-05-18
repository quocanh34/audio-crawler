import datasets 
from datasets import Dataset 

# dataset1 = datasets.load_dataset("quocanh34/final_dataset_wer0_v1")
dataset2 = datasets.load_dataset("quocanh34/final_dataset_wer0_v2")
dataset3 = datasets.load_dataset("quocanh34/final_dataset_wer0_v3")

from datasets import Dataset, concatenate_datasets
my_dataset = {
    "audio": [],
    "transcription": [],
    "w2v2_transcription": [],
    "WER": []
}

final_dataset_wer0_400hrs = Dataset.from_dict(my_dataset)
# final_dataset_wer0_500hrs = concatenate_datasets([final_dataset_wer0_500hrs, dataset1['train']])
final_dataset_wer0_400hrs = concatenate_datasets([final_dataset_wer0_400hrs, dataset2['train']])
final_dataset_wer0_400hrs = concatenate_datasets([final_dataset_wer0_400hrs, dataset3['train']])


def calculate_hour(dataset):
    # Calculate the total duration of the dataset in seconds 
    total_duration_seconds = sum([len(audio["array"]) / audio["sampling_rate"] for audio in dataset["audio"]]) 
    
    # Convert the total duration to hours 
    total_duration_hours = total_duration_seconds / 3600 
    return total_duration_hours

print(final_dataset_wer0_400hrs)

# print("The total duration of the final dataset WER0 (predicted 400hrs) is: " + str(calculate_hour(final_dataset_wer0_400hrs)) + " hours.") 

final_dataset_wer0_400hrs.push_to_hub("quocanh34/final_dataset_wer0_400hrs", token="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")