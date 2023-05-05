
import datasets 
from datasets import Dataset 

dataset1 = datasets.load_dataset("quocanh34/youtube_dataset_split1_vid_300")
dataset2 = datasets.load_dataset("quocanh34/youtube_dataset_split8_final")
dataset3 = datasets.load_dataset("quocanh34/youtube_dataset_new1_vid_1000")
dataset4 = datasets.load_dataset("quocanh34/youtube_dataset_new3_vid_1000")
dataset5 = datasets.load_dataset("quocanh34/youtube_dataset_new5_final")
dataset6 = datasets.load_dataset("quocanh34/youtube_dataset_new2_vid_500")
dataset7 = datasets.load_dataset("quocanh34/youtube_dataset_new4_final")
dataset8 = datasets.load_dataset("quocanh34/youtube_dataset_new_cut_final")

from datasets import Dataset, concatenate_datasets
my_dataset = {
    "audio": [],
    "transcription": [],
    "w2v2_transcription": [],
    "WER": []
}

final_dataset_wer10_v2 = Dataset.from_dict(my_dataset)
final_dataset_wer10_v3 = Dataset.from_dict(my_dataset)

final_dataset_wer10_v2 = concatenate_datasets([final_dataset_wer10_v2, dataset2['train']])
final_dataset_wer10_v2 = concatenate_datasets([final_dataset_wer10_v2, dataset3['train']])
final_dataset_wer10_v2 = concatenate_datasets([final_dataset_wer10_v2, dataset5['train']])
final_dataset_wer10_v2 = concatenate_datasets([final_dataset_wer10_v2, dataset8['train']])

final_dataset_wer10_v3 = concatenate_datasets([final_dataset_wer10_v3, dataset1['train']])
final_dataset_wer10_v3 = concatenate_datasets([final_dataset_wer10_v3, dataset4['train']])
final_dataset_wer10_v3 = concatenate_datasets([final_dataset_wer10_v3, dataset6['train']])
final_dataset_wer10_v3 = concatenate_datasets([final_dataset_wer10_v3, dataset7['train']])


def filter_wer(dataset):
    return dataset['WER'] == 0

final_dataset_wer0_v2 = final_dataset_wer10_v2.filter(filter_wer)
final_dataset_wer0_v3 = final_dataset_wer10_v3.filter(filter_wer)


def calculate_hour(dataset):
    # Calculate the total duration of the dataset in seconds 
    total_duration_seconds = sum([len(audio["array"]) / audio["sampling_rate"] for audio in dataset["audio"]]) 
    
    # Convert the total duration to hours 
    total_duration_hours = total_duration_seconds / 3600 
    return total_duration_hours

  
print("The total duration of the dataset v2 WER = 10 is: " + str(calculate_hour(final_dataset_wer10_v2)) + " hours.") 
print("The total duration of the dataset v2 WER = 0 is: " + str(calculate_hour(final_dataset_wer0_v2)) + " hours.") 
print("*"*20)
print("The total duration of the dataset v3 WER = 10 is: " + str(calculate_hour(final_dataset_wer10_v3)) + " hours.") 
print("The total duration of the dataset v3 WER = 0 is: " + str(calculate_hour(final_dataset_wer0_v3)) + " hours.") 


final_dataset_wer0_v2.push_to_hub("quocanh34/final_dataset_wer0_v2", token="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")
final_dataset_wer0_v3.push_to_hub("quocanh34/final_dataset_wer0_v3", token="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")

final_dataset_wer10_v2.push_to_hub("quocanh34/final_dataset_wer10_v2", token="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")
final_dataset_wer10_v3.push_to_hub("quocanh34/final_dataset_wer10_v3", token="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")


