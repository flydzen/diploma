from typing import Optional, Iterator, Set, Tuple
from itertools import islice

import torch
import wget
import zipfile
from contextlib import redirect_stderr, redirect_stdout
import shutil
import sys
from opentypesvg import fonts2svg
from pathlib import Path
import pandas as pd
from typing import List

import numpy as np
from tqdm import tqdm
from git import Repo
import random
from copy import deepcopy
from torch.utils.data import Dataset

from utils.svg import SVG

URL = 'https://github.com/duskvirkus/dafonts-free/releases/download/v1.0.0/dafonts-free-v1.zip'
GGL_FONTS_GIT = 'https://github.com/google/fonts.git'
OUT_PATH_ROOT = (cur.parent if (cur := Path('.').absolute()).parent == 'utils' else cur) / 'data'
OUT_PATH_ZIP = OUT_PATH_ROOT / 'fonts.zip'
OUT_PATH_GGL = OUT_PATH_ROOT / 'ggl'
OUT_PATH_PROCESSED = OUT_PATH_ROOT.parent.parent / 'diploma' / 'data' / 'svg'
OUT_PATH_ENCODED = OUT_PATH_ROOT / 'encoded'

OUT_PATH_ENCODED_TEST = OUT_PATH_ENCODED / 'test'
OUT_PATH_ENCODED_TRAIN = OUT_PATH_ENCODED / 'train'

GLYPH_FILTER = list(map(chr, range(ord('a'), ord('z') + 1))) + \
               list(map(chr, range(ord('A'), ord('Z') + 1))) + \
               ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def _bar_custom(current, total, width=80):
    """
    Прогрессбар для загрузчика wget
    """
    width = max(40, width)
    current += 1
    p = int(current / total * width)
    progress_message = f'\r<{"#" * p + "_" * (width - p)}>: {str(current / total * 100)[:4]}% [{current} / {total}]'
    if current % 10 == 0:
        sys.stdout.write(progress_message)
        sys.stdout.flush()


def _download() -> None:
    """
    Скачивает датасет dafonts-free-v1 из URL в OUT_PATH_ZIP
    """
    OUT_PATH_ROOT.mkdir(parents=True, exist_ok=True)

    if OUT_PATH_ZIP.exists():
        return
    print('Downloading...')
    wget.download(URL, out=str(OUT_PATH_ZIP), bar=_bar_custom)

    print('Extracting...')

    with zipfile.ZipFile(OUT_PATH_ZIP, 'r') as file:
        file.extractall(OUT_PATH_ROOT)

    print('Extracted')


def _clone() -> None:
    """
    Клонирует гит с гугловским датасетом
    """
    if not OUT_PATH_GGL.exists():
        print('Cloning...')
        Repo.clone_from(GGL_FONTS_GIT, OUT_PATH_GGL)
        print('Cloned')


def _get_blacklist() -> Set[str]:
    """
    Возвращает множество плохих шрифтов
    """
    with Path(OUT_PATH_ROOT / 'blacklist.txt').open('r') as file:
        fonts = set(map(str.strip, file.readlines()))
    return fonts


def _ttf_to_svg(file, out_dir_suffix) -> Optional[Path]:
    """
    Распаковывает шрифт в набор svg файлов.
    :param file: Путь до файла шрифта (.ttf, .otf)
    :param out_dir_suffix: суффикс для папки распаковки. Имя шрифта
    :return: Path до папки с svg, если удалось распаковать, None иначе
    """
    out = OUT_PATH_PROCESSED / out_dir_suffix
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        try:
            with open('logs', 'a') as logs:
                with redirect_stderr(logs), redirect_stdout(logs):
                    fonts2svg.main(['-u', '-o', str(out), str(file)])
        except Exception:
            shutil.rmtree(out)
            return None
        return out
    return None


def _ttf_dir_to_svg(directory: Path, filters=None, size=None) -> None:
    """
    Распаковывает все шрифты из папки в svg файлы, оставляет только нужные буквы.
    :param directory: Папка со шрифтами
    :param filters: Список имен файлов, которые оставит (названия глифов)
    :param size: Количество шрифтов, верхняя граница.
    :return:
    """
    print('Converting fonts to svg...')
    filters = filters or GLYPH_FILTER

    items = list(directory.rglob('*.otf')) + list(directory.rglob('*.ttf'))

    blacklist = _get_blacklist()
    clear_items = []
    for item in items:
        if item.stem in blacklist:
            item.unlink(True)
        else:
            clear_items.append(item)
    items = clear_items
    for i, file in tqdm(enumerate(items[:size]), total=size or len(items)):
        font_name = file.stem
        if '.' in font_name:  # кто так называет?
            file.unlink()
            continue
        out = font_name
        if processed := _ttf_to_svg(file, out):
            matched = _fonts_filter(processed, filters)
            for item in matched:
                try:
                    svg = SVG.load(item)
                    svg.prepare()
                    svg.dump_to_file()
                except Exception:
                    item.unlink(True)
    print('Converted')


def _fonts_filter(font_directory: Path, filters=None) -> List[Path]:
    """
    Фильтрует svg файлы в папке по имени, не прошедшие фильтрацию удаляет.
    Все svg файлы в поддиректориях переносит в `font_directory`
    :param font_directory: Папка с svg файлами
    :param filters: Имена файлов
    :return: Список файлов, соответствующих фильтру
    """
    filters = filters or GLYPH_FILTER
    filter_svg = [f'{i}.svg' for i in filters]
    matched = []
    for file in font_directory.rglob('*.svg'):
        if str(file.name) not in filter_svg:
            file.unlink()
        else:
            matched.append(file)
    return matched


def _get_font_paths() -> Iterator[Path]:
    """
    Ищет папки с svg файлами = шрифты
    :return: Итератор путей до папок
    """
    data_dir = OUT_PATH_PROCESSED
    if data_dir.exists():
        fonts = list(data_dir.iterdir())
        for font in fonts:
            yield font.absolute()
    return


def load_data(size=None) -> None:
    """
    Скачивает данные и конвертирует их в папки с svg файлами.
    :param size: Ограничение на кол-во шрифтов. Из двух датасетов будет пополам
    """
    half_size = size // 2 if size else None
    _download()
    _clone()
    _ttf_dir_to_svg(OUT_PATH_ROOT / 'dafonts-free-v1/fonts', size=half_size)
    _ttf_dir_to_svg(OUT_PATH_GGL, size=half_size)


def encode_data(size=None, test_size: float = 0.1, augment=True):
    """
    Берет готовые svg и кодирует их для создания датасета. Также создает аугментацию.
    В OUT_PATH_ENCODED / test.svg и OUT_PATH_ENCODED / train.svg лежат документы с указанием
     пути файла с соответствующими лейблом и шрифтом.
    То, куда указывает путь - numpy массив с закодированной буквой.
    :param size:
    :param test_size:
    :param augment:
    :return:
    """

    def save_labels(_is_test, _font_name, _letter, _out_path):
        if _is_test:
            labels_test.append((_font_name, _letter, _out_path))
        else:
            labels_train.append((_font_name, _letter, _out_path))

    def get_save_path(_is_test: bool, _font_name: str, _letter: str) -> Path:
        if _letter.isupper():
            _letter = f'({_letter})'
        return (OUT_PATH_ENCODED_TEST if is_test else OUT_PATH_ENCODED_TRAIN) / _font_name / f'{_letter}.npy'

    def process_svg(_svg: SVG, _out_path, _is_test, _font_name, _letter) -> bool:
        encoded = _svg.encode()
        if encoded is not None:
            _out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(_out_path, encoded)
            save_labels(_is_test=_is_test, _font_name=_font_name, _letter=_letter, _out_path=_out_path)
            return True
        return False

    def do_augment(
            _is_test: bool,
            _font_name: str,
            _letter: str,
            original: SVG,
            w: float,  # 0.8, 1.2
    ):
        if not augment:
            return
        aug_font_name = f'{_font_name}_augS{str(w).replace(".", "x")}'
        aug_out_path = get_save_path(_is_test=_is_test, _font_name=aug_font_name, _letter=_letter)

        svg_copy = deepcopy(original)
        svg_copy.stretch(w)
        process_svg(
            _svg=svg_copy, _out_path=aug_out_path, _is_test=_is_test, _font_name=aug_font_name, _letter=_letter
        )

    print('Encoding...')
    # if OUT_PATH_ENCODED.exists():
    #     print('Data already exists')
    #     return

    OUT_PATH_ENCODED.mkdir(parents=True, exist_ok=True)
    labels_test = []
    labels_train = []
    size = min(size or 1e10, len(list(_get_font_paths())))
    for path in tqdm(islice(_get_font_paths(), size), total=size):
        font_name = path.stem
        is_test = hash(font_name) % 1000 < test_size * 1000

        to_aug = []
        for glyf_path in path.rglob('*.svg'):
            letter = glyf_path.stem

            out_path = get_save_path(_is_test=is_test, _font_name=font_name, _letter=letter)
            if out_path.exists():
                continue

            svg = SVG.load(glyf_path)
            if len(svg.commands) > SVG.ENCODE_HEIGHT:
                continue
            if process_svg(_svg=svg, _out_path=out_path, _is_test=is_test, _font_name=font_name, _letter=letter):
                to_aug.append(dict(_is_test=is_test, _font_name=font_name, _letter=letter, original=svg))
        for kwargs in to_aug:
            do_augment(**kwargs, w=0.8)
        for kwargs in to_aug:
            do_augment(**kwargs, w=1.2)

    labels_test_df = pd.DataFrame(data=labels_test, columns=['font', 'letter', 'path'])
    labels_train_df = pd.DataFrame(data=labels_train, columns=['font', 'letter', 'path'])
    labels_test_df.to_csv(OUT_PATH_ENCODED / 'test.csv')
    labels_train_df.to_csv(OUT_PATH_ENCODED / 'train.csv')
    print('Encoded')
    
    
def collate_fn(items):
    dts, lbls, f_names = [], [], []
    for dt, lbl, f_name in items:
        dts.append(dt)
        lbls.append(lbl)
        f_names.append(f_name)
    
    return dts, lbls, f_names


class FontsDataset(Dataset):
    def __init__(self, test=False, download=False, download_size=None):
        file_name = 'test' if test else 'train'
        data_csv = OUT_PATH_ENCODED / f'{file_name}.csv'
        data_npy = OUT_PATH_ENCODED / f'{file_name}.npy'
        
        if download and not data_csv.exists() and not data_npy.exists():
            load_data(size=download_size)
            encode_data(size=download_size)
        info = pd.read_csv(data_csv)
        if data_npy.exists():
            self.data = np.load(data_npy)
        else:
            self.data = np.zeros((len(info), SVG.ENCODE_HEIGHT, SVG.ENCODE_WIDTH), dtype=np.float32)
            for i in tqdm(range(len(info))):
                self.data[i] = np.load(info.iloc[i, 3])
            np.save(data_npy, self.data)
            
        self.font_names = []
        self.glyphs = []
        self.letters = []
        
        for index, row in tqdm(info.iterrows(), total=len(info), desc='Grouping by font'):
            index: int = index
            if not self.font_names or row['font'] != self.font_names[-1]:
                if index != 0:
                    self.glyphs.append(self.data[index - len(self.letters[-1]):index])
                self.font_names.append(row['font'])
                self.letters.append([])

            self.letters[-1].append(row['letter'])
        self.glyphs.append(self.data[index - len(self.letters[-1]):index])
        
    def __len__(self):
        return len(self.font_names)

    def __getitem__(self, idx) -> Tuple[List[np.ndarray], List[str], List[str]]:
        return self.glyphs[idx], self.letters[idx], [self.font_names[idx]] * len(self.letters[idx])
