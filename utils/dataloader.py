from typing import Optional, Iterator
from itertools import islice

import wget
import zipfile
from contextlib import redirect_stderr, redirect_stdout
import shutil
import sys
from opentypesvg import fonts2svg
from pathlib import Path
import pandas as pd
from typing import List
from tqdm import tqdm
from git import Repo

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


def bar_custom(current, total, width=80):
    width = max(40, width)
    current += 1
    p = int(current / total * width)
    progress_message = f'\r<{"#" * p + "_" * (width - p)}>: {str(current / total * 100)[:4]}% [{current} / {total}]'
    if current % 10 == 0:
        sys.stdout.write(progress_message)
        sys.stdout.flush()


def download() -> None:
    """
    Скачивает датасет dafonts-free-v1 из URL в OUT_PATH_ZIP
    """
    OUT_PATH_ROOT.mkdir(parents=True, exist_ok=True)

    if OUT_PATH_ZIP.exists():
        return
    print('Downloading...')
    wget.download(URL, out=str(OUT_PATH_ZIP), bar=bar_custom)

    print('Extracting...')

    with zipfile.ZipFile(OUT_PATH_ZIP, 'r') as file:
        file.extractall(OUT_PATH_ROOT)

    print('Extracted')


def clone() -> None:
    if not OUT_PATH_GGL.exists():
        print('Cloning...')
        Repo.clone_from(GGL_FONTS_GIT, OUT_PATH_GGL)
        print('Cloned')


def get_blacklist() -> set[str]:
    with Path(OUT_PATH_ROOT / 'blacklist.txt').open('r') as file:
        fonts = set(map(str.strip, file.readlines()))
    return fonts


def ttf_to_svg(file, out_dir_suffix) -> Optional[Path]:
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
        return ttf_to_svg(file, out_dir_suffix)
    return None


def ttf_dir_to_svg(directory: Path, filters=None, size=None) -> None:
    filters = filters or GLYPH_FILTER

    items = list(directory.rglob('*.ttf'))

    blacklist = get_blacklist()
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
        if processed := ttf_to_svg(file, out):
            matched = fonts_filter(processed, filters)
            for item in matched:
                try:
                    svg = SVG.load(item)
                    svg.prepare()
                    svg.dump_to_file()
                except Exception:
                    item.unlink(True)
    print()


def fonts_filter(font_directory: Path, filters=None) -> List[Path]:
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


def get_font_paths() -> Iterator[Path]:
    data_dir = Path('data/svg')
    if data_dir.exists():
        fonts = list(data_dir.iterdir())
        for font in fonts:
            yield font.absolute()
    return


def load_data(size=None):
    half_size = size // 2 if size else None
    download()
    ttf_dir_to_svg(OUT_PATH_ROOT / 'dafonts-free-v1/fonts', size=half_size)
    clone()
    ttf_dir_to_svg(OUT_PATH_GGL, size=half_size)


def get_data(size=None):
    data = []
    size = size or len(list(get_font_paths()))
    for i, path in tqdm(enumerate(islice(get_font_paths(), size)), total=size):
        name = path.name
        for glif in path.iterdir():
            letter = glif.stem
            svg = SVG.load(glif)
            if len(svg.commands) > SVG.ENCODE_HEIGHT:
                continue
            encoded = svg.encode()
            if encoded is not None:
                data.append((name, letter, encoded))
    return pd.DataFrame(data=data, columns=['font', 'letter', 'data'])


def encode_data(size=None):
    data = []
    size = size or len(list(get_font_paths()))
    for i, path in tqdm(enumerate(islice(get_font_paths(), size)), total=size):
        name = path.name
        for glif in path.iterdir():
            letter = glif.stem
            svg = SVG.load(glif)
            if len(svg.commands) > SVG.ENCODE_HEIGHT:
                continue
            encoded = svg.encode()
            if encoded is not None:
                data.append((name, letter, encoded))
    return pd.DataFrame(data=data, columns=['font', 'letter', 'data'])


