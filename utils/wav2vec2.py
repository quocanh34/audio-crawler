import os
import zipfile
import soundfile as sf
import torch
import kenlm

from transformers.file_utils import cached_path, hf_bucket_url
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

class Wav2Vec2():
    def __init__(self, cache_path, wav2vec2_path):
        self.cache_path = cache_path
        self.wav2vec2_path = wav2vec2_path

    def get_processor(self):
        processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_path, cache_dir=self.cache_path)
        self.processor = processor

    def get_model(self):
        model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_path, cache_dir=self.cache_path)
        self.model = model
    
    def get_lm_file(self):
        lm_file = hf_bucket_url(self.wav2vec2_path, filename='vi_lm_4grams.bin.zip')
        lm_file = cached_path(lm_file, cache_dir=self.cache_path)
        with zipfile.ZipFile(lm_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_path)
        lm_file = self.cache_path + 'vi_lm_4grams.bin'
        self.lm_file = lm_file
    
    def get_decoder_ngram_model(self):
        vocab_dict = self.processor.tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:-2]
        vocab_list = vocab

        # convert ctc blank character representation
        vocab_list[self.processor.tokenizer.pad_token_id] = ""

        # replace special characters
        vocab_list[self.processor.tokenizer.unk_token_id] = ""

        # convert space character representation
        vocab_list[self.processor.tokenizer.word_delimiter_token_id] = " "

        # specify ctc blank char index, since conventially it is the last entry of the logit matrix
        alphabet = Alphabet.build_alphabet(
            vocab_list, ctc_token_idx=self.processor.tokenizer.pad_token_id)
        lm_model = kenlm.Model(self.lm_file)
        decoder = BeamSearchDecoderCTC(alphabet,
                                    language_model=LanguageModel(lm_model))
        self.ngram_lm_model = decoder
        # return self.ngram_lm_model


    def add_w2v2_label(self, example):

        ds = {}
        ds["file"] = example['audio']['path']
        ds["speech"] = example['audio']['array']  # array
        ds["sampling_rate"] = example['audio']['sampling_rate']

        # infer model
        input_values = self.processor(
            ds["speech"],
            sampling_rate=ds["sampling_rate"],
            return_tensors="pt"
        ).input_values.to("cuda")

        self.model.to("cuda")

        logits = self.model(input_values).logits[0]
        # print(logits.shape)

        # decode ctc output
        pred_ids = torch.argmax(logits, dim=-1)
        # greedy_search_output = processor.decode(pred_ids)
        beam_search_output = self.ngram_lm_model.decode(
            logits.cpu().detach().numpy(), beam_width=500)

        # Compute the new value
        new_label = beam_search_output

        # Add the new column to the example
        example['w2v2_transcription'] = new_label

        # Empty cuda
        del input_values
        del logits
        del pred_ids
        torch.cuda.empty_cache()

        # Return the modified example
        return example