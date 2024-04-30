import os
from operator import itemgetter

import numpy as np
from speechbrain.dataio.dataio import read_audio


def get_samples(root):
    audios = []
    references = []
    for book in os.listdir(root):
        for chapter in os.listdir(f"{root}/{book}"):
            for file in os.listdir(f"{root}/{book}/{chapter}"):
                if file.endswith("txt"):
                    with open(f"{root}/{book}/{chapter}/{file}", "r") as f:
                        for line in f.readlines():
                            audio_path, reference = line.split(" ", 1)
                            full_audio_path = (
                                f"{root}/{book}/{chapter}/{audio_path}.flac"
                            )
                            audios.append(read_audio(full_audio_path))
                            references.append(reference)
    return audios, references


def random_choice(items, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(len(items), n)
    return list(itemgetter(*indices)(items))
