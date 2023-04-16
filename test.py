import datasets
from datasets import Dataset
# Load your dataset
# dataset = datasets.load_dataset("quocanh34/youtube_dataset_v2")

# # Calculate the total duration of the dataset in seconds
# total_duration_seconds = sum([len(audio["array"]) / audio["sampling_rate"] for audio in dataset["train"]["audio"]])

# # Convert the total duration to hours
# total_duration_hours = total_duration_seconds / 3600

# print(f"The total duration of the dataset is {total_duration_hours:.2f} hours.")

# print(dataset)

my_dataset = {
    "audio": [],
    "transcription": [],
    "w2v2_transcription": [],
    "WER": []
}

final_dataset = Dataset.from_dict(my_dataset)
print(final_dataset)