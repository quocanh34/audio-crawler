import datasets
import os
import sys
import torch

from predict import load_model


def audio_align(example):
    wav = torch.tensor(example['audio']['array']).float()
    wav = wav.view(1, -1)
    rate = example['audio']['sampling_rate']
    


    return

def main(data_links):
    # Load data
    data = datasets.load_dataset(data_links)

    # Load model
    load_model()
    
    data.map(audio_align)


if __name__ == '__main__':
    main(sys.argv[1])