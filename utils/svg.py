import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import xmltodict


COMMANDS = 'lcp'


class EncodeError(Exception):
    pass


class SVG:
    OFFSET = 0.05
    ONE_HOT_LEN = 5
    ENCODE_HEIGHT = 64
    ENCODE_WIDTH = ONE_HOT_LEN + 6
    ENCODE_SHAPE = (ENCODE_WIDTH, ENCODE_HEIGHT)

    def __init__(self, commands: list, view_box: tuple = None, file: Path = None):
        self.file = file
        self.commands = commands
        self.view_box = view_box
        self.relative: Optional[bool] = None

    def simplify(self):
        assert self.relative is None
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        new_commands = []
        last = [0, 0]
        for command in self.commands:
            match command:
                case 'M' | 'L' | 'C' as q, *args:
                    new_arg = [q, *args]
                    last = args[-2:]
                case 'm' | 'l' as q, x, y:  # move | line
                    last[0] += x
                    last[1] += y
                    new_arg = [q.upper(), *last]
                case 'H', x:  # horizontal line
                    last[0] = x
                    new_arg = ['L', *last]
                case 'h', x:  # relative horizontal line
                    last[0] += x
                    new_arg = ['L', *last]
                case 'V', y:  # vertical line
                    last[1] = y
                    new_arg = ['L', *last]
                case 'v', y:  # relative vertical line
                    last[1] += y
                    new_arg = ['L', *last]
                case 'c', *arg:  # cubic curve
                    new_arg = ['C', *[arg[i] + last[i % 2] for i in range(6)]]
                    last = new_arg[-2:]
                case 'Q' | 'q' as q, px, py, x, y:  # quadratic curve, conversion to cubic
                    if q == 'q':
                        px += last[0]
                        py += last[1]
                        x += last[0]
                        y += last[1]
                    new_arg = [
                        'C',
                        last[0] + 2 / 3 * (px - last[0]), last[1] + 2 / 3 * (py - last[1]),
                        x + 2 / 3 * (px - x), y + 2 / 3 * (py - y),
                        x, y,
                    ]
                    last = [x, y]
                case 'Z' | 'z', :
                    new_arg = ['Z']
                case _ as unsupported:
                    raise Exception(f'Unsupported command {unsupported}, path: {self.file}')
            min_x = min(min_x, last[0])
            min_y = min(min_y, last[1])
            max_x = max(max_x, last[0])
            max_y = max(max_y, last[1])
            new_commands.append(new_arg)
        self.view_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        self.commands = new_commands
        self.relative = False

    def to_relative(self):
        def shift(arguments):
            nonlocal last
            last[0] += arguments[-2]
            last[1] += arguments[-1]

        assert self.relative is False

        new_commands = []
        start = [0, 0]
        last = [0, 0]
        for command in self.commands:
            match command:
                case 'M' as q, x, y:  # move
                    new_arg = [q.lower(), x - last[0], y - last[1]]
                    last = [x, y]
                    start = last.copy()
                case 'L' as q, x, y:  # line
                    new_arg = [q.lower(), x - last[0], y - last[1]]
                    last = [x, y]
                case 'C', *arg:  # cubic curve
                    new_arg = ['c', *[arg[i] - last[i % 2] for i in range(6)]]
                    shift(new_arg)
                case 'Z' | 'z',:
                    new_arg = ['z']
                    last = start.copy()
                case _ as unsupported:
                    raise Exception(f'Unsupported command {unsupported}')
            new_commands.append(new_arg)
        self.commands = new_commands
        self.relative = True

    def normalize(self):
        assert self.view_box is not None
        if self.relative is False:
            for c in self.commands:
                for i, ch in enumerate(c[1:]):
                    c[i + 1] = ch - self.view_box[i % 2]

            scale = max(self.view_box[2:])
            for command in self.commands:
                for i, v in enumerate(command[1:], start=1):
                    command[i] = v / scale
        elif self.relative is True:
            x_max, x_min, y_max, y_min = float('-inf'), float('inf'), float('-inf'), float('inf')
            last = self.commands[0][-2:]
            start = last
            for c in self.commands[1:]:
                if c[0] == 'z':
                    last = start
                    continue
                min_x = min(min_x, last[0])
                min_y = min(min_y, last[1])
                max_x = max(max_x, last[0])
                max_y = max(max_y, last[1])
                last[0] += c[-2]
                last[1] += c[-1]
                if c[0] == 'm':
                    start = last
            scale = max(x_max - x_min, y_max - y_min)
            
            self.commands[0][-2] -= x_min
            self.commands[0][-1] -= y_min
            for c in self.commands:
                for i in range(1, len(c)):
                    c[i] /= scale
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
            commands.append([str(command[0])] + [np.format_float_positional(i, precision=8, trim='-') for i in command[1:]])
        path = '\t' + '\n\t'.join([' '.join(command) for command in commands])
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
            'm': 1,
            'l': 2,
            'c': 3,
            'z': 4,
        }
        result = []

        start = np.array([0, 0], dtype=np.float32)
        last = start.copy()

        for command, *args in self.commands:
            line = np.zeros(self.ENCODE_WIDTH, dtype=np.float32)
            args = np.array(args, dtype=np.float32)

            if command == 'z':
                last = start.copy()
            else:
                last += args[-2:]
            if command == 'm':
                start = last.copy()

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
        one_hot_match = ' mlcz'
        commands = []
        for row in data:
            command = one_hot_match[np.argmax(row[:SVG.ONE_HOT_LEN])]
            args = row[SVG.ONE_HOT_LEN:]
            line = [command]
            match command:
                case 'm' | 'l':
                    line.extend(args[-2:])
                case 'z':
                    pass
                case 'c':
                    line.extend(args)
                case ' ':
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
        self.to_relative()


if __name__ == '__main__':
    svg = SVG.load(Path('../data/svg/!paulmaul/o.svg'))

    svg.prepare()

    svg.dump_to_file(Path('1.svg'))

    encoded = svg.encode()
    decoded = SVG.decode(encoded)

    decoded.dump_to_file(Path('2.svg'))

#     # svg = SVG.load(Path('../data/svg/dafonts/!sketchytimes/six.svg'))
#     svg.simplify()
#     svg.normalize()
#     print(svg.encode())
#     svg.dump_to_file(Path('../testing.svg'))
