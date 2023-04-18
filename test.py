import datasets
from datasets import Dataset
# Load your dataset
test_dataset = datasets.load_dataset("quocanh34/youtube_dataset_v2")

# Calculate the total duration of the dataset in seconds
total_duration_seconds = sum([len(audio["array"]) / audio["sampling_rate"] for audio in test_dataset["train"]["audio"]])

# Convert the total duration to hours
total_duration_hours = total_duration_seconds / 3600

print(f"The total duration of the dataset is {total_duration_hours:.2f} hours.")

print(test_dataset)
