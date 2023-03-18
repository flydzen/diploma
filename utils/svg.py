import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import xmltodict


class EncodeError(Exception):
    pass


class SVG:
    OFFSET = 0.05
    ONE_HOT_LEN = 5
    ENCODE_HEIGHT = 80
    ENCODE_WIDTH = ONE_HOT_LEN + 6
    ENCODE_SHAPE = (ENCODE_WIDTH, ENCODE_HEIGHT)

    def __init__(self, commands: list, view_box: tuple = None, file: Path = None):
        self.file = file
        self.commands = commands
        self.view_box = view_box or (0, 0, 1, 1)
        self.relative: Optional[bool] = None

    def simplify(self):
        assert self.relative is None
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        new_commands = []
        last = [0, 0]
        for command in self.commands:
            if command[0] == 'Z':
                new_command = command
            elif command[0] in ('M', 'L', 'C'):
                new_command = command
                last = command[-2:]
            elif command[0] == 'H':
                last[0] = command[1]
                new_command = ['L', *last]
            elif command[0] == 'V':
                last[1] = command[1]
                new_command = ['L', *last]
            elif command[0] == 'Q':
                q, px, py, x, y = command
                new_command = [
                    'C',
                    last[0] + 2 / 3 * (px - last[0]), 
                    last[1] + 2 / 3 * (py - last[1]),
                    x + 2 / 3 * (px - x), 
                    y + 2 / 3 * (py - y),
                    x, y,
                ]
                last = [x, y]
            else:
                raise Exception(f'Unsupported command {command}, path: {self.file}')

            for i, c in enumerate(new_command[1:], start=1):
                if i % 2 == 1:  # x
                    min_x = min(min_x, c)
                    max_x = max(max_x, c)
                else:  # y
                    min_y = min(min_y, c)
                    max_y = max(max_y, c)

            new_commands.append(new_command)
        self.view_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        self.commands = new_commands

    def normalize(self):
        assert self.view_box is not None
        for c in self.commands:
            for i, ch in enumerate(c[1:]):
                c[i + 1] = ch - self.view_box[i % 2]

        scale = max(self.view_box[2:])
        for command in self.commands:
            for i, v in enumerate(command[1:], start=1):
                command[i] = v / scale
        self.view_box = (0, 0, 1, 1)

    @classmethod
    def load(cls, file: Path):
        with open(str(file), 'r') as f:
            doc = xmltodict.parse(f.read())
            path = doc['svg']['path']['@d']
            res = list(
                filter(
                    lambda y: y != '',
                    [''.join(x).strip() for _, x in itertools.groupby(path, key=lambda x: x if x.isalpha() else False)],
                )
            )
            command = None
            commands = []
            for d in res:
                if d[0].isalpha():
                    if command is not None:
                        commands.append([command])
                    command = d
                else:
                    commands.append([command, *list(map(float, d.split()))])
                    command = None
            if command is not None:
                commands.append([command])
        return SVG(commands, file=file)

    def dump(self):
        view_box = ' '.join(map(str, self.view_box)) if self.view_box else (0, 0, 1, 1)
        commands = []
        for command in self.commands:
            commands.append(
                [str(command[0])] + [np.format_float_positional(i, precision=7, trim='-') for i in command[1:]]
            )
        path = '\n'.join([' '.join(command) for command in commands])
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">\n' \
               f'    <path fill="#000000" d="\n' \
               f'{path}\n' \
               f'    "/>\n' \
               f'</svg>\n'
    
    def dump_to_file(self, file: Path = None):
        file = file or self.file
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('w') as f:
            f.write(self.dump())

    def encode(self):
        if len(self.commands) > self.ENCODE_HEIGHT - 1:
            return None

        one_hot_match = {
            ' ': 0,
            'M': 1,
            'L': 2,
            'C': 3,
            'Z': 4,
        }
        result = []

        for command, *args in self.commands:
            line = np.zeros(self.ENCODE_WIDTH, dtype=np.float32)
            args = np.array(args, dtype=np.float32)

            assert command in one_hot_match, f'Wrong command {command}, file {self.file}'

            line[one_hot_match[command]] = 1.
            if len(args) > 0:
                line[-len(args):] = args
            result.append(line)

        for _ in range(len(result), SVG.ENCODE_HEIGHT):
            line = np.zeros(self.ENCODE_WIDTH)
            line[one_hot_match[' ']] = 1.
            result.append(line)
        return np.array(result, dtype=np.float32)

    @classmethod
    def decode(cls, data, path: Path = None):
        one_hot_match = ' MLCZ'
        commands = []
        for row in data:
            command = one_hot_match[np.argmax(row[:SVG.ONE_HOT_LEN])]
            args = row[SVG.ONE_HOT_LEN:]
            line = [command]
            if command == 'M' or command == 'L':
                line.extend(args[-2:])
            elif command == 'Z':
                pass
            elif command == 'C':
                line.extend(args)
            else:
                break
            commands.append(line)
        svg = SVG(commands, view_box=(0, 0, 1, 1), file=path)
        svg.relative = True
        return svg
    
    def mulsize(self, x):
        for command in self.commands:
            for v in range(1, len(command)):
                command[v] *= x
        self.view_box = (0, 0, self.view_box[2] * x, self.view_box[3] * x) 

    def prepare(self):
        self.simplify()
        self.normalize()
