from jiwer import wer

def filter_wer(dataset):
    return dataset['WER'] <= 10 
