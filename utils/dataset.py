import pandas as pd
import datasets
from datasets.features import Audio
from datasets import load_dataset


class DatasetOperations():
    def __init__(self, path_to_csv, new_path_to_csv, data_folder):
        self.path_to_csv = path_to_csv
        self.new_path_to_csv = new_path_to_csv
        self.data_folder = data_folder
        self.min_label_length = 5
        self.min_audio_length = 1.5

    def create_new_csv(self):
        df = pd.read_csv(self.path_to_csv, delimiter='|', header=None)
        df.columns = ['file_name', 'transcription']
        df.to_csv(self.new_path_to_csv)

    def create_dataset(self):
        self.dataset = load_dataset("audiofolder", data_dir=self.data_folder)
        return self.dataset

    def remove_column(self):
        self.dataset = self.dataset.remove_columns("Unnamed: 0")
        return self.dataset

    def cast_column(self):
        self.dataset = self.dataset.cast_column(
            "audio", Audio(sampling_rate=16000))
        return self.dataset

    def is_alphanumeric(self, labels):
        return all(c.isalnum() or c.isspace() for c in labels)

    def is_longer_than_min_label_length(self, labels):
        return len(labels.split()) > self.min_label_length

    def is_longer_than_min_audio_length(self, audios):
        return (len(audios["array"]) / audios["sampling_rate"]) > self.min_audio_length

    def filter_non_characters(self):
        self.dataset = self.dataset.filter(
            self.is_alphanumeric, input_columns=['transcription'])
        return self.dataset

    def filter_labels(self):
        self.dataset = self.dataset.filter(
            self.is_longer_than_min_label_length, input_columns=['transcription'])
        return self.dataset

    def filter_audios(self):
        self.dataset = self.dataset.filter(
            self.is_longer_than_min_audio_length, input_columns=['audio'])
        return self.dataset

    def lower_labels(self, example):
        example['transcription'] = example['transcription'].lower()
        return example

    def normalize(self):
        self.dataset = self.dataset.map(self.lower_labels)
        return self.dataset

    def print_dataset(self):
        return self.dataset
