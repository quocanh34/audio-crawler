import argparse
from datasets import load_dataset

def filter_wer0(dataset):
    return dataset['WER'] == 0

def calculate_wer0_hours(dataset):
    wer0_dataset = dataset.filter(filter_wer0)
    # Calculate the total duration of the dataset in seconds
    total_duration_seconds = sum([len(audio["array"]) / audio["sampling_rate"] for audio in wer0_dataset["train"]["audio"]])

    # Convert the total duration to hours
    total_duration_hours = total_duration_seconds / 3600

    print(f"The total duration of the dataset WER = 0 is {total_duration_hours:.2f} hours.")

def calculate_wer10_hours(dataset):

    # Calculate the total duration of the dataset in seconds
    total_duration_seconds = sum([len(audio["array"]) / audio["sampling_rate"] for audio in dataset["train"]["audio"]])

    # Convert the total duration to hours
    total_duration_hours = total_duration_seconds / 3600

    print(f"The total duration of the dataset WER <= 10 is {total_duration_hours:.2f} hours.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate total hours of audio data from Hugging Face dataset')
    parser.add_argument('--dataset_path', type=str, help='Path to the Hugging Face dataset')
    parser.add_argument('--wer', type=int, help='Word Error Rate threshold')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    
    if args.wer == 0:
        calculate_wer0_hours(dataset)
    elif args.wer == 10:
        calculate_wer10_hours(dataset)
    