from typing import Optional, Iterator
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
OUT_PATH_PROCESSED = OUT_PATH_ROOT / 'svg'
OUT_PATH_ENCODED = OUT_PATH_ROOT / 'encoded'

OUT_PATH_ENCODED_TEST = OUT_PATH_ENCODED / 'test'
OUT_PATH_ENCODED_TRAIN = OUT_PATH_ENCODED / 'train'

GLYPH_FILTER = list(map(chr, range(ord('a'), ord('z') + 1))) + \
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


def _get_blacklist() -> set[str]:
    """
    Возвращает множество плохих шрифтов
    """
    with Path(OUT_PATH_ROOT / 'blacklist.txt').open('r') as file:
        fonts = set(map(str.strip, file.readlines()))
    return fonts


def _ttf_to_svg(file, out_dir_suffix) -> Optional[Path]:
    """
    Распаковывает шрифт в набор svg файлов.
    На второй запуск проверяет, что все получилось, и если нет, пробует починить.
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
    elif (out / '_moreSVGs_').exists():
        shutil.rmtree(out)
        return _ttf_to_svg(file, out_dir_suffix)
    return None


def _ttf_dir_to_svg(directory: Path, filters=None, size=None) -> None:
    """
    Распаковывает все шрифты из папки в svg файлы, оставляет только нужные буквы.
    :param directory: Папка со шрифтами
    :param filters: Список имен файлов, которые оставит (названия глифов)
    :param size: Количество шрифтов, верхняя граница.
    :return:
    """
    filters = filters or GLYPH_FILTER

    items = list(directory.rglob('*.ttf'))

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
    print()


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
            matched.append(font_directory / file.name)
            file.rename(font_directory / file.name)
    for d in font_directory.iterdir():
        if d.is_dir():
            shutil.rmtree(d)
    return matched


def _get_font_paths() -> Iterator[Path]:
    """
    Ищет папки с svg файлами = шрифты
    :return: Итератор путей до папок
    """
    data_dir = Path('data/svg')
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
    _ttf_dir_to_svg(OUT_PATH_ROOT / 'dafonts-free-v1/fonts', size=half_size)
    _clone()
    _ttf_dir_to_svg(OUT_PATH_GGL, size=half_size)


def encode_data(size=None, test_size: float = 0.1, augment=True):
    """
    Берет готовые svg и кодирует их для создания датасета. Также создает аугментацию
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
            original: SVG
    ):
        if not augment:
            return
        for w in (0.85, 1.15):
            aug_font_name = f'{_font_name}_augS{str(w).replace(".", "x")}'
            aug_out_path = get_save_path(_is_test=_is_test, _font_name=aug_font_name, _letter=_letter)

            svg_copy = deepcopy(original)
            svg_copy.stretch(w)
            process_svg(
                    _svg=svg_copy, _out_path=aug_out_path, _is_test=_is_test, _font_name=aug_font_name, _letter=_letter
            )

    if OUT_PATH_ENCODED.exists():
        print('Data already exists')
        return

    labels_test = []
    labels_train = []
    size = size or len(list(_get_font_paths()))
    for path in tqdm(islice(_get_font_paths(), size), total=size):
        font_name = path.stem
        is_test = random.uniform(0, 1) < test_size

        for glif_path in path.iterdir():
            letter = glif_path.stem

            out_path = get_save_path(_is_test=is_test, _font_name=font_name, _letter=letter)

            svg = SVG.load(glif_path)
            if process_svg(_svg=svg, _out_path=out_path, _is_test=is_test, _font_name=font_name, _letter=letter):
                do_augment(_is_test=is_test, _font_name=font_name, _letter=letter, original=svg)

    labels_test_df = pd.DataFrame(data=labels_test, columns=['font', 'letter', 'path'])
    labels_train_df = pd.DataFrame(data=labels_train, columns=['font', 'letter', 'path'])
    labels_test_df.to_csv(OUT_PATH_ENCODED / 'test.csv')
    labels_train_df.to_csv(OUT_PATH_ENCODED / 'train.csv')


class FontsDataset(Dataset):
    def __init__(self, test=False, download=False, download_size=None):
        if download and not OUT_PATH_ENCODED.exists():
            load_data(size=download_size)
            encode_data(size=download_size)
        self.info = pd.read_csv(OUT_PATH_ENCODED / ('test.csv' if test else 'train.csv'))

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx) -> tuple[np.ndarray, str, str]:
        font_name = self.info.iloc[idx, 1]
        letter = self.info.iloc[idx, 2]
        data = np.load(self.info.iloc[idx, 3])
        return data, letter, font_name
