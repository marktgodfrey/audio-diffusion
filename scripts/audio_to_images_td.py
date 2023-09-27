import os
import re
import logging
import argparse
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, Features, Array2D, Value
import librosa

MAX_SHARD_SIZE = 8192

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger('audio_to_images')


def main(args):
    sample_rate = args.sample_rate
    audio_files = [
        os.path.join(root, file) for root, _, files in os.walk(args.input_dir)
        for file in files if re.search(r"\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    shard_count = 0
    examples = []
    for audio_file in tqdm(audio_files):
        try:
            audio, _ = librosa.load(audio_file, mono=True, sr=sample_rate)
        except KeyboardInterrupt:
            raise
        except Exception as exp:
            print(audio_file)
            print(exp)
            continue

        num_slices = len(audio) // args.win_length
        for i in range(num_slices):
            y = audio[(args.win_length * i):(args.win_length * (i + 1))]
            if len(y) < args.win_length:
                print('slice too short?')
            else:
                y = np.expand_dims(y, axis=0)
                examples.append({
                    "audio": y.tolist(),
                    "audio_file": audio_file,
                    "slice": i,
                })

        if len(examples) >= MAX_SHARD_SIZE:
            ds = Dataset.from_list(
                examples,
                features=Features({
                    "audio": Array2D(shape=(1, args.win_length), dtype='float32'),
                    "audio_file": Value(dtype="string"),
                    "slice": Value(dtype="int16"),
                }),
            )
            dsd = DatasetDict({"train": ds})
            out_path = os.path.join(args.output_dir, '%d' % shard_count)
            os.makedirs(out_path, exist_ok=True)
            dsd.save_to_disk(out_path)
            print('saved: %s' % out_path)
            shard_count += 1
            examples = []

    if len(examples) > 0:
        ds = Dataset.from_list(
            examples,
            features=Features({
                "audio": Array2D(shape=(1, args.win_length), dtype='float32'),
                "audio_file": Value(dtype="string"),
                "slice": Value(dtype="int16"),
            }),
        )
        dsd = DatasetDict({"train": ds})
        out_path = os.path.join(args.output_dir, '%d' % shard_count)
        os.makedirs(out_path, exist_ok=True)
        dsd.save_to_disk(out_path)
        print('saved: %s' % out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Create dataset of audio slices from directory of audio files.")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--win_length", type=int, default=16384)
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    if args.input_dir is None:
        raise ValueError(
            "You must specify an input directory for the audio files.")

    main(args)
