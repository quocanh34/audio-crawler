import os
import pandas as pd
import datasets
import shutil
import dotenv
import torch.multiprocessing as mp
import torch

from vctube import VCtube
from youtube_transcript_api import YouTubeTranscriptApi
from datasets import Dataset, concatenate_datasets
from jiwer import wer

from utils.dataset import DatasetOperations
from utils.wav2vec2 import Wav2Vec2
from utils.wer import filter_wer

def main():
    config_env = dotenv.dotenv_values(".env")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    current_dir = os.getcwd()
    youtube_df = pd.read_csv(current_dir + config_env["CSV_LINK"])

    my_dataset = {
        "audio": [],
        "transcription": [],
        "w2v2_transcription": [],
        "WER": []
    }

    final_dataset = Dataset.from_dict(my_dataset)

    for index, row in youtube_df.iterrows():
        try:
            ctx = mp.get_context('spawn')
            q = ctx.Queue()
            p = ctx.Process(target=process_dataset, args=(row, config_env, current_dir, q))
            p.start()
            dataset = q.get()
            p.join()

            if dataset is None:
                print("Skipped Index: " + str(index+1))
                push_dataset(final_dataset=final_dataset, config_env=config_env, index=index)
                shutil.rmtree(current_dir + config_env["DATA_FILE"])
                continue

            if dataset is not None:
                final_dataset = concatenate_datasets([final_dataset, dataset])

                print("Current index: " + str(index+1))
                print(final_dataset)

                push_dataset(final_dataset=final_dataset, config_env=config_env, index=index)
                shutil.rmtree(current_dir + config_env["DATA_FILE"])
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in row {index+1}: {e}")
            print(f"Error in link: {row}")

            push_dataset(final_dataset=final_dataset, config_env=config_env, index=index)
            shutil.rmtree(current_dir + config_env["DATA_FILE"])
            torch.cuda.empty_cache()
            continue
    
    push_dataset(final_dataset, config_env)
    print(final_dataset)

def process_dataset(row, config_env, current_dir, q):

    path_to_data_files = current_dir + config_env["DATA_FILE"]
    path_to_csv = path_to_data_files + config_env["META_DATA"]
    new_path_to_csv = path_to_data_files + config_env["NEW_META_DATA"]
    data_folder = path_to_data_files + config_env["DATA_FOLDER"]

    youtube_link = row["youtube link"]

    vc = VCtube(path_to_data_files, youtube_link, lang='vi')
    if (vc.check_vi_available()):
        vc.operations()
    else:
        return None

    operations = DatasetOperations(path_to_csv, new_path_to_csv, data_folder)
    operations.create_new_csv()
    dataset = operations.create_dataset()
    dataset = operations.remove_column()
    dataset = operations.cast_column()

    dataset = operations.filter_non_characters()
    dataset = operations.filter_labels()
    dataset = operations.filter_audios()
    dataset = operations.normalize()

    wav2vec2 = Wav2Vec2(cache_path=config_env["CACHE_PATH"], wav2vec2_path=config_env["WAV2VEC2_PATH"])
    wav2vec2.get_processor()
    wav2vec2.get_model()
    wav2vec2.get_lm_file()
    wav2vec2.get_decoder_ngram_model()

    dataset = dataset['train'].map(lambda example: wav2vec2.add_w2v2_label(example), num_proc=4)
    dataset = dataset.map(lambda example: {"WER": int(wer(example["transcription"], example["w2v2_transcription"]) * 100)})
    dataset = dataset.filter(filter_wer)

    #Empty cuda cache
    torch.cuda.empty_cache()

    q.put(dataset)

    return dataset

def push_dataset(final_dataset, config_env, index=None):

    if index != None and (index+1) % 4 == 0:
        final_dataset.push_to_hub(config_env["HUGGINGFACE_HUB"] + f"_test_vid_{index+1}", token=config_env["TOKEN"])
        print("-"*10)
        print(f"Dataset vid_{index+1} has been pushed to hub!")
        print("-"*10)
    elif index == None:
        final_dataset.push_to_hub(config_env["HUGGINGFACE_HUB"] +"_test_final", token=config_env["TOKEN"])
    else:
        pass

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()