import torch
import torchaudio
import string
import pandas as pd
import math
import csv
import sys 
import os
from typing import List, Tuple
from torch import distributions
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


from utils import TextTransform

class TrainDataset(torch.utils.data.Dataset):
    """Custom competition dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.answers = pd.read_csv(csv_file, '\t')
        self.transform = transform


    def __len__(self):
        return len(self.answers)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        utt_name = 'cv-corpus-5.1-2020-06-22/en/clips/' + self.answers.loc[idx, 'path']
        utt = torchaudio.load(utt_name)[0].squeeze()
        if len(utt.shape) != 1:
            utt = utt[1]

        answer = self.answers.loc[idx, 'sentence']

        if self.transform:
            utt = self.transform(utt)

        sample = {'utt': utt, 'answer': answer}
        return sample


class TestDataset(torch.utils.data.Dataset):
    """Custom test dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.names = pd.read_csv(csv_file, '\t')
        self.transform = transform


    def __len__(self):
        return len(self.names)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        utt_name = 'cv-corpus-5.1-2020-06-22/en/clips/' + self.names.loc[idx, 'path']
        utt = torchaudio.load(utt_name)[0].squeeze()

        if self.transform:
            utt = self.transform(utt)

        sample = {'utt': utt}
        return sample


#win_len=1024, hop_len=256
# counting len of MelSpec before doing it (cause of padding)
def mel_len(x):
    return int(x // 256) + 1


def transform_tr(wav):
    aug_num = torch.randint(low=0, high=3, size=(1,)).item()
    augs = [
        lambda x: x,
        lambda x: (x + distributions.Normal(0, 0.01).sample(x.size())).clamp_(-1, 1),
        lambda x: torchaudio.transforms.Vol(.1)(x)
    ]
    return augs[aug_num](wav)


# collate_fn
def preprocess_data(data):
    text_transform = TextTransform()
    wavs = []
    input_lens = []
    labels = []
    label_lens = []

    for el in data:
        wavs.append(el['utt'])
        input_lens.append(math.ceil(mel_len(el['utt'].shape[0]) / 2))    # cause of stride 2
        label = torch.Tensor(text_transform.text_to_int(el['answer']))
        labels.append(label)
        label_lens.append(len(label))

    wavs = pad_sequence(wavs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    return wavs, input_lens, labels, label_lens



def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples

    Arguments:
        unicode_csv_data: unicode csv data (see example below)

    Examples:
        >>> from torchtext.utils import unicode_csv_reader
        >>> import io
        >>> with io.open(data_path, encoding="utf8") as f:
        >>>     reader = unicode_csv_reader(f)

    """

    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    for line in csv.reader(unicode_csv_data, **kwargs):
        yield line
        
        
        




def load_commonvoice_item(line: List[str],
                          header: List[str],
                          path: str,
                          folder_audio: str,
                          tsv: str,
                          tr):
    # Each line as the following data:
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent

    assert header[1] == "path"
    fileid = line[1]
    filename = os.path.join(path.replace("/" + tsv, ""), folder_audio, fileid)
    #filename = "datasets/CommonVoice/cv-corpus-12.0-delta-2022-12-07-en/cv-corpus-12.0-delta-2022-12-07/en/clips/common_voice_en_34925857.mp3"
    utt = torchaudio.load(str(filename), format = "mp3")[0].squeeze()
    if(tr):
        utt = tr(utt)
    dic = dict(zip(header, line))
    
    sample = {'utt': utt, 'answer': dic['sentence'].lower().replace(",", "").replace(".", "").replace(":", "").replace("-", "").replace('"', "").replace("(", "").replace(")", "")}
    return sample


class COMMONVOICE(Dataset):
    """Create a Dataset for CommonVoice.

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        tsv (str, optional): The name of the tsv file used to construct the metadata.
            (default: ``"train.tsv"``)
        url (str, optional): The URL to download the dataset from, or the language of
            the dataset to download. (default: ``"english"``).
            Allowed language values are ``"tatar"``, ``"english"``, ``"german"``,
            ``"french"``, ``"welsh"``, ``"breton"``, ``"chuvash"``, ``"turkish"``, ``"kyrgyz"``,
            ``"irish"``, ``"kabyle"``, ``"catalan"``, ``"taiwanese"``, ``"slovenian"``,
            ``"italian"``, ``"dutch"``, ``"hakha chin"``, ``"esperanto"``, ``"estonian"``,
            ``"persian"``, ``"portuguese"``, ``"basque"``, ``"spanish"``, ``"chinese"``,
            ``"mongolian"``, ``"sakha"``, ``"dhivehi"``, ``"kinyarwanda"``, ``"swedish"``,
            ``"russian"``, ``"indonesian"``, ``"arabic"``, ``"tamil"``, ``"interlingua"``,
            ``"latvian"``, ``"japanese"``, ``"votic"``, ``"abkhaz"``, ``"cantonese"`` and
            ``"romansh sursilvan"``.
        folder_in_archive (str, optional): The top-level directory of the dataset.
        version (str): Version string. (default: ``"cv-corpus-4-2019-12-10"``)
            For the other allowed values, Please checkout https://commonvoice.mozilla.org/en/datasets.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".txt"
    _ext_audio = ".mp3"
    _folder_audio = "clips"

    def __init__(self,
                 root="datasets/CommonVoice/cv-corpus-12.0-delta-2022-12-07-en/cv-corpus-12.0-delta-2022-12-07/en",
                 tsv = "invalidated.tsv",
                 url = "None",
                 folder_in_archive = "None",
                 version = "None",
                 download = "None",
                 transforms =None):

        languages = {
            "tatar": "tt",
            "english": "en",
            "german": "de",
            "french": "fr",
            "welsh": "cy",
            "breton": "br",
            "chuvash": "cv",
            "turkish": "tr",
            "kyrgyz": "ky",
            "irish": "ga-IE",
            "kabyle": "kab",
            "catalan": "ca",
            "taiwanese": "zh-TW",
            "slovenian": "sl",
            "italian": "it",
            "dutch": "nl",
            "hakha chin": "cnh",
            "esperanto": "eo",
            "estonian": "et",
            "persian": "fa",
            "portuguese": "pt",
            "basque": "eu",
            "spanish": "es",
            "chinese": "zh-CN",
            "mongolian": "mn",
            "sakha": "sah",
            "dhivehi": "dv",
            "kinyarwanda": "rw",
            "swedish": "sv-SE",
            "russian": "ru",
            "indonesian": "id",
            "arabic": "ar",
            "tamil": "ta",
            "interlingua": "ia",
            "latvian": "lv",
            "japanese": "ja",
            "votic": "vot",
            "abkhaz": "ab",
            "cantonese": "zh-HK",
            "romansh sursilvan": "rm-sursilv"
        }



        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, version, basename)

        self._path = os.path.join(root, tsv)

        

        self._tsv = os.path.join(root, tsv)
        self.tsv = tsv
        self.tr = transforms

        with open(self._tsv, "r") as tsv:
            walker = unicode_csv_reader(tsv, delimiter="\t")
            self._header = next(walker)
            self._walker = list(walker)

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, dictionary)``,  where dictionary is built
            from the TSV file with the following keys: ``client_id``, ``path``, ``sentence``,
            ``up_votes``, ``down_votes``, ``age``, ``gender`` and ``accent``.
        """
        line = self._walker[n]
        
        return load_commonvoice_item(line, self._header, self._path, self._folder_audio, self.tsv, self.tr)


    def __len__(self) -> int:
        return len(self._walker)
    
    
    
    