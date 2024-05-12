import os
from operator import itemgetter

import numpy as np
from speechbrain.dataio.dataio import read_audio


def get_librispeech_data(root):
    """Gets audio samples and references from downloaded LibriSpeech data.
    Assumes that the data is downloaded, and the directory structure matches LibriSpeech.
    Download by:
    ```
    mkdir librispeech_dev_clean
    wget https://www.openslr.org/resources/12/dev-clean.tar.gz -P /content
    tar -xvf dev-clean.tar.gz -C librispeech_dev_clean
    ```

    Arguments
    ---------
    root : str
        Path to root of directory

    Returns
    -------
    tuple[list[torch.Tensor], list[str]]
        tuple of list of audio tensors and list of corresponding reference texts
    """
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
    """Randomly choose n items from iterable.

    Arguments
    ---------
    items : Iterable
    n : int
        number of items to choose
    seed : int
        numpy seed for reproducability

    Returns
    -------
    list of chosen items
    """
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(len(items), n)
    return list(itemgetter(*indices)(items))
