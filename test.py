# import datasets
# from datasets import Dataset
# # Load your dataset
# test_dataset = datasets.load_dataset("quocanh34/youtube_dataset_v2")

# # Calculate the total duration of the dataset in seconds
# total_duration_seconds = sum([len(audio["array"]) / audio["sampling_rate"] for audio in test_dataset["train"]["audio"]])

# # Convert the total duration to hours
# total_duration_hours = total_duration_seconds / 3600

# print(f"The total duration of the dataset is {total_duration_hours:.2f} hours.")

# print(test_dataset)
import datasets 
from datasets import Dataset 

dataset1 = datasets.load_dataset("quocanh34/youtube_datasetsplit1_v2_final")
dataset2 = datasets.load_dataset("quocanh34/youtube_datasetsplit2_v2_final")
dataset3 = datasets.load_dataset("quocanh34/youtube_datasetsplit3_v3_final")
dataset4 = datasets.load_dataset("quocanh34/youtube_dataset_split1_1000_vid_250")
dataset5 = datasets.load_dataset("quocanh34/youtube_dataset_split_2_vid_1200")
dataset6 = datasets.load_dataset("quocanh34/youtube_datasetsplit3_vid_700")
dataset7 = datasets.load_dataset("quocanh34/youtube_datasetsplit4_vid_1000")
dataset8 = datasets.load_dataset("quocanh34/youtube_datasetsplit5_vid_1000")
dataset9 = datasets.load_dataset("quocanh34/youtube_datasetsplit6_vid_1000")
dataset10 = datasets.load_dataset("quocanh34/youtube_datasetsplit7_vid_1000")
dataset11 = datasets.load_dataset("quocanh34/youtube_dataset_hope_vid_300")
dataset12 = datasets.load_dataset("quocanh34/youtube_datasetexpress_vid_350")
dataset13 = datasets.load_dataset("quocanh34/youtube_dataset_zlife_final")

from datasets import Dataset, concatenate_datasets
my_dataset = {
    "audio": [],
    "transcription": [],
    "w2v2_transcription": [],
    "WER": []
}

final_dataset_wer10_v1 = Dataset.from_dict(my_dataset)
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset1['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset2['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset3['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset4['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset5['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset6['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset7['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset8['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset9['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset10['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset11['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset12['train']])
final_dataset_wer10_v1 = concatenate_datasets([final_dataset_wer10_v1, dataset13['train']])

print(final_dataset_wer10_v1)

def filter_wer(dataset):
    return dataset['WER'] == 0

final_dataset_wer0_v1 = final_dataset_wer10_v1.filter(filter_wer)
print(final_dataset_wer0_v1)

def calculate_wer10_hour():
    # Calculate the total duration of the dataset in seconds 
    total_duration_seconds_1 = sum([len(audio["array"]) / audio["sampling_rate"] for audio in final_dataset_wer10_v1["audio"]]) 
    
    # Convert the total duration to hours 
    total_duration_hours_1 = total_duration_seconds_1 / 3600 
    return total_duration_hours_1

def calculate_wer0_hour():
    # Calculate the total duration of the dataset in seconds 
    total_duration_seconds_2 = sum([len(audio["array"]) / audio["sampling_rate"] for audio in final_dataset_wer0_v1["audio"]]) 
    
    # Convert the total duration to hours 
    total_duration_hours_2 = total_duration_seconds_2 / 3600 
    return total_duration_hours_2

  
print("The total duration of the dataset WER = 10 is: " + str(calculate_wer10_hour) + " hours.") 
print("The total duration of the dataset WER = 0 is: " + str(calculate_wer0_hour) + " hours.") 


final_dataset_wer10_v1.push_to_hub("quocanh34/final_dataset_wer10_v1", token="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")
final_dataset_wer0_v1.push_to_hub("quocanh34/final_dataset_wer0_v1", token="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")