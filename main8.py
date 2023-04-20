import os
import pandas as pd
import datasets
import shutil
import dotenv

from vctube import VCtube
from youtube_transcript_api import YouTubeTranscriptApi
from datasets import Dataset, concatenate_datasets
from jiwer import wer

from utils.dataset import DatasetOperations
from utils.wav2vec2 import Wav2Vec2
from utils.wer import filter_wer

def main():
    # load .env config
    config_env = dotenv.dotenv_values(".env")

    #set environment for cuda
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    #get current path
    current_dir = os.getcwd()
    # print(config_env)

    # read youtube csv file
    youtube_df = pd.read_csv(current_dir + config_env["CSV_LINK8"])

    my_dataset = {
        "audio": [],
        "transcription": [],
        "w2v2_transcription": [],
        "WER": []
    }

    final_dataset = Dataset.from_dict(my_dataset)

    for index, row in youtube_df.iterrows():
        try:
            path_to_data_files = current_dir + config_env["DATA_FILE"]
            path_to_csv = path_to_data_files + config_env["META_DATA"]
            new_path_to_csv = path_to_data_files + config_env["NEW_META_DATA"]
            data_folder = path_to_data_files + config_env["DATA_FOLDER"]

            youtube_link = row["youtube link"]

            # Use VCTube to download, split, remove YT videos into audio and transcription
            vc = VCtube(path_to_data_files, youtube_link, lang='vi')
            if (vc.check_vi_available()):
                vc.operations()
            else:
                continue
            
            
            # Transform above data in to Huggingface Dataset
            operations = DatasetOperations(path_to_csv, new_path_to_csv, data_folder)
            operations.create_new_csv()
            dataset = operations.create_dataset()
            dataset = operations.remove_column()
            dataset = operations.cast_column()

            # Filtering data in Dataset
            dataset = operations.filter_non_characters()
            dataset = operations.filter_labels()
            dataset = operations.filter_audios()
            dataset = operations.normalize()

            # Wav2Vec2 result using VCTube splited audio
            wav2vec2 = Wav2Vec2(cache_path=config_env["CACHE_PATH"], wav2vec2_path=config_env["WAV2VEC2_PATH"])
            wav2vec2.get_processor()
            wav2vec2.get_model()
            wav2vec2.get_lm_file()
            wav2vec2.get_decoder_ngram_model()

            dataset = dataset['train'].map(
                lambda example: wav2vec2.add_w2v2_label(example), num_proc=4)

            # Caculate WER between Wav2Vec2 and VCTube transcription
            dataset = dataset.map(lambda example: {"WER": int(
                wer(example["transcription"], example["w2v2_transcription"]) * 100)})
            
            dataset = dataset.filter(filter_wer)

            # Concatenate the dataset to final dataset
            final_dataset = concatenate_datasets([final_dataset, dataset])
            
            #print current data info
            print("Current index: " + str(index+1))
            print(final_dataset)

            #push to huggingface if the index is multiple numbers of 1000
            if (index+1) % 10 == 0:
                final_dataset.push_to_hub(config_env["HUGGINGFACE_HUB"] + f"split8_vid_{index+1}", token=config_env["TOKEN"])
                print("-"*10)
                print(f"Dataset vid_{index+1} has been pushed to hub!")
                print("-"*10)

            #Delete data files, ready for the next loop
            shutil.rmtree(path_to_data_files)
        except Exception as e:
            print(f"Error in row {index+1}: {e}")
            print(f"Error in row {row}")

    final_dataset.push_to_hub(config_env["HUGGINGFACE_HUB"] +"split8_final", token=config_env["TOKEN"])
    print(final_dataset)
main()