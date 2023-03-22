import os
import re
import io
import logging
import argparse

MAX_SHARD_SIZE = 8192

import numpy as np
# import pandas as pd
from tqdm.auto import tqdm
from diffusers.pipelines.audio_diffusion import Mel
from datasets import Dataset, DatasetDict, Features, Array2D, Value

import librosa

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger('audio_to_images')


def main(args):
    mel = Mel(x_res=args.resolution[0],
              y_res=args.resolution[1],
              hop_length=args.hop_length,
              sample_rate=args.sample_rate,
              n_fft=args.n_fft)
    audio_files = [
        os.path.join(root, file) for root, _, files in os.walk(args.input_dir)
        for file in files if re.search(r"\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    shard_count = 0
    examples = []
    for audio_file in tqdm(audio_files):
        try:
            mel.load_audio(audio_file)
        except KeyboardInterrupt:
            raise
        except:
            continue
        for slice in range(mel.get_number_of_slices()):
            y = mel.get_audio_slice(slice)
            melspec = librosa.feature.melspectrogram(
                y=y,
                sr=mel.sr,
                n_fft=mel.n_fft,
                hop_length=mel.hop_length,
                n_mels=mel.n_mels
            )
            melspec_db = librosa.power_to_db(melspec, ref=np.max, top_db=mel.top_db)
            melspec_db_norm = ((melspec_db + mel.top_db) / mel.top_db).clip(0., 1.)
            examples.append({
                "melspec": melspec_db_norm.tolist(),
                "audio_file": audio_file,
                "slice": slice,
            })

            # image = mel.audio_slice_to_image(slice)
            # assert (image.width == args.resolution[0] and image.height
            #         == args.resolution[1]), "Wrong resolution"
            # # skip completely silent slices
            # if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
            #     logger.warning('File %s slice %d is completely silent',
            #                    audio_file, slice)
            #     continue
            # with io.BytesIO() as output:
            #     image.save(output, format="PNG")
            #     bytes = output.getvalue()
            # examples.extend([{
            #     "image": {
            #         "bytes": bytes
            #     },
            #     "audio_file": audio_file,
            #     "slice": slice,
            # }])
        if len(examples) >= MAX_SHARD_SIZE:
            # ds = Dataset.from_pandas(
            #     pd.DataFrame(examples),
            #     features=Features({
            #         # "image": Image(),
            #         "melspec": Array2D(shape=(args.resolution[1], args.resolution[0]), dtype='float32'),
            #         "audio_file": Value(dtype="string"),
            #         "slice": Value(dtype="int16"),
            #     }),
            # )
            ds = Dataset.from_list(
                examples,
                features=Features({
                    # "image": Image(),
                    "melspec": Array2D(shape=(args.resolution[1], args.resolution[0]), dtype='float32'),
                    "audio_file": Value(dtype="string"),
                    "slice": Value(dtype="int16"),
                }),
            )
            dsd = DatasetDict({"train": ds})
            out_path = os.path.join(args.output_dir, '%d' % shard_count)
            os.makedirs(out_path, exist_ok=True)
            dsd.save_to_disk(out_path)
            if args.push_to_hub:
                dsd.push_to_hub(args.push_to_hub)
            print('saved: %s' % out_path)
            shard_count += 1
            examples = []

    if len(examples) > 0:
        ds = Dataset.from_list(
            examples,
            features=Features({
                # "image": Image(),
                "melspec": Array2D(shape=(args.resolution[1], args.resolution[0]), dtype='float32'),
                "audio_file": Value(dtype="string"),
                "slice": Value(dtype="int16"),
            }),
        )
        dsd = DatasetDict({"train": ds})
        out_path = os.path.join(args.output_dir, '%d' % shard_count)
        os.makedirs(out_path, exist_ok=True)
        dsd.save_to_disk(out_path)
        if args.push_to_hub:
            dsd.push_to_hub(args.push_to_hub)
        print('saved: %s' % out_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Create dataset of Mel spectrograms from directory of audio files.")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--resolution",
                        type=str,
                        default="256",
                        help="Either square resolution or width,height.")
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    args = parser.parse_args()

    if args.input_dir is None:
        raise ValueError(
            "You must specify an input directory for the audio files.")

    # Handle the resolutions.
    try:
        args.resolution = (int(args.resolution), int(args.resolution))
    except ValueError:
        try:
            args.resolution = tuple(int(x) for x in args.resolution.split(","))
            if len(args.resolution) != 2:
                raise ValueError
        except ValueError:
            raise ValueError(
                "Resolution must be a tuple of two integers or a single integer."
            )
    assert isinstance(args.resolution, tuple)

    main(args)
