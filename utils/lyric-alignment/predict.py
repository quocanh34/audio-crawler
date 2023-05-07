import os
from transformers import AutoTokenizer, AutoFeatureExtractor
from model_handling import Wav2Vec2ForCTC
import torch
import datasets
import utils
import sys
from tqdm import tqdm
import torchaudio
import json
import time
import csv
import argparse

use_gpu = True
if use_gpu:
    if not torch.cuda.is_available():
        use_gpu = False
model_path = 'nguyenvulebinh/lyric-alignment'
model = None
tokenizer = None
feature_extractor = None
vocab = None

def load_model():
    global model
    global tokenizer
    global feature_extractor
    global vocab

    model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
    if use_gpu:
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer.get_vocab()))]


def do_asr(waveform):
    input_values = feature_extractor.pad([
        {"input_values": feature_extractor(item, sampling_rate=16000)["input_values"][0]} for item in
        waveform
    ], return_tensors='pt')["input_values"]
    if use_gpu:
        input_values = input_values.cuda()

    out_values = model(input_values=input_values)
    logits = out_values.logits[0]

    emissions = torch.log_softmax(logits, dim=-1)
    emission = emissions.cpu().detach()
    emission[emission < -20] = -20

    emission[:,tokenizer.convert_tokens_to_ids('|')] = torch.tensor([max(item) for item in list(zip(emission[:,tokenizer.convert_tokens_to_ids('|')], emission[:,tokenizer.convert_tokens_to_ids('<pad>')]))])
    emission[:,tokenizer.convert_tokens_to_ids('<pad>')] = -20

    return emission

def do_force_align(waveform, emission, word_list, sample_rate=16000, base_stamp=0):
    transcript = '|'.join(word_list)
    dictionary = {c: i for i, c in enumerate(vocab)}
    tokens = [dictionary.get(c, 0) for c in transcript]
    trellis = utils.get_trellis(emission, tokens, blank_id=tokenizer.convert_tokens_to_ids('|'))
    path = utils.backtrack(trellis, emission, tokens)
    segments = utils.merge_repeats(path, transcript)
    word_segments = utils.merge_words(segments)
    word_segments = utils.add_pad(word_segments, emission)
    ratio = waveform.size(1) / (trellis.size(0) - 1)
    result = []
    for word in word_segments:
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)

        result.append({
            'd': word.label,
            's': int(x0/sample_rate*1000)+base_stamp,
            'e': int(x1/sample_rate*1000)+base_stamp,
            'score': word.score,
            'x0': x0,
            'x1': x1,
        })

    assert [item['d'] for item in result] == word_list

    return result

def audio_align(example, padding):
    wav = torch.tensor(example['audio']['array']).float()
    wav = wav.view(1, -1)

    words = example['w2v2_transcription'].split()
    single_words_list = [word.split() for word in words]
    single_words = [y for x in single_words_list for y in x]

    # wav to text_prob
    emission = do_asr(wav)
    # text_prob to (single_words & timestamp)
    word_piece = do_force_align(wav, emission, single_words)

    audio_start = word_piece[0]['x0']
    audio_end = word_piece[-1]['x1']
    example['audio']['array'] = example['audio']['array'][audio_start:audio_end + padding]
    return example


def main(data_links, output_path, num_workers, padding):
    # Load data
    data = datasets.load_dataset(data_links)

    # Load model
    load_model()

    aligned_data = data.map(audio_align, fn_kwargs = {"padding", padding}, num_proc = num_workers)
    aligned_data.push_to_hub(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_links', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--padding', type=int, default=1000)
    args = parser.parse_args()
    main(args.data_links, args.output_path, args.num_workers, args.padding)