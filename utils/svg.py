import itertools
import math
from pathlib import Path
import numpy as np
import xmltodict


class EncodeError(Exception):
    pass


class SVG:
    ONE_HOT_LEN = 5
    ENCODE_HEIGHT = 32
    ENCODE_WIDTH = ONE_HOT_LEN + 6
    ENCODE_SHAPE = (ENCODE_WIDTH, ENCODE_HEIGHT)

    def __init__(self, commands: list, view_box: tuple = None, file: Path = None, height=None, width=None):
        self.file = file
        self.commands = commands
        self.view_box = view_box or (0, 0, 1, 1)
        self.h = height or 256
        self.w = width or 256

    def simplify(self):
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

    def rearrange(self):
        components = []
        last = -1
        for i, line in enumerate(self.commands):
            if line[0] == 'Z':
                components.append(self.commands[last + 1: i + 1])
                last = i

        components = [self.rearrange_component(component) for component in components]
        components.sort(key=lambda x: x[0][-2:])
        self.commands = list(itertools.chain(*components))

    @staticmethod
    def rearrange_component(component):
        main_commands = component[1:-1]

        # команда Z фиктивна. Завершаем путь в ту же стартовую точку вручную, если не завершен
        if component[0][-2:] != main_commands[-1][-2:]:
            main_commands.append(['L', *component[0][-2:]])

        points = [c[-2:] for c in main_commands]
        start_point = min(points)
        start_index = points.index(start_point)
        rearranged_commands = [
            ['M', *start_point],
            *main_commands[start_index + 1:],
            *main_commands[:start_index + 1],
            ['Z'],
        ]
        return rearranged_commands

    @classmethod
    def load(cls, file: Path):
        with open(str(file), 'r') as f:
            doc = xmltodict.parse(f.read())
            path = doc['svg']['path']['@d']
            height = doc['svg']['@height']
            width = doc['svg']['@width']
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
        return SVG(commands, file=file, height=height, width=width)

    def dump(self):
        view_box = ' '.join(map(str, self.view_box)) if self.view_box else (0, 0, 1, 1)
        commands = []
        for command in self.commands:
            commands.append(
                [str(command[0])] + [np.format_float_positional(i, precision=7, trim='-') for i in command[1:]]
            )
        path = '\n'.join([' '.join(command) for command in commands])
        return f'<svg xmlns="http://www.w3.org/2000/svg" ' \
               f'viewBox="{view_box}" height="{self.h}px" width="{self.w}px">\n' \
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
        return svg
    
    def mulsize(self, x):
        for command in self.commands:
            for v in range(1, len(command)):
                command[v] *= x
        self.view_box = (0, 0, self.view_box[2] * x, self.view_box[3] * x) 

    def stretch(self, x):
        for command in self.commands:
            for v in range(1, len(command), 2):
                command[v] *= x
        self.view_box = (0, 0, self.view_box[2], self.view_box[3] * x)
        self.normalize()

    def prepare(self):
        self.simplify()
        self.rearrange()
        self.normalize()

    def refine(self):
        MIN_NORM = 0.005
        angle_limit = math.radians(10)

        p_c, p_args, p_out = None, None, None
        to_remove = []
        repeat = True
        while repeat:
            repeat = False
            for i, (c, *args) in enumerate(self.commands):
                args = np.array(args)
                if p_c is None:
                    p_c, p_args = c, args
                    continue
                nxt = self.commands[(i + 1) % len(self.commands)]
                if c == 'Z':
                    nxt[-2] = p_args[-1]
                    nxt[-1] = p_args[-2]
                    continue
                norm = np.linalg.norm(args[-2:] - p_args[-2:])
                if norm < MIN_NORM:
                    to_remove.append(i)
                    continue

                if c == 'L' or c == 'M':
                    if abs(args[-2] / args[-1]) < 0.01:
                        args[-2] = 0
                    if abs(args[-1] / args[-2]) < 0.01:
                        args[-1] = 0
                    in_vec = args[-2:] - p_args[-2:]
                    in_vec /= np.linalg.norm(in_vec)
                    out_vec = in_vec
                    out_vec /= np.linalg.norm(out_vec)

                    dot_product = np.dot(p_out, in_vec)
                    angle = np.arccos(dot_product)

                else:
                    in_vec = args[:2] - p_args[-2:]
                    in_norm = np.linalg.norm(in_vec)
                    in_vec /= in_norm
                    out_vec = (args[-2:] - args[-4:-2])
                    out_vec /= np.linalg.norm(out_vec)

                    dot_product = np.dot(p_out, in_vec)
                    angle = np.arccos(dot_product)
                    if np.abs(angle) < 10:
                        args[:2] = p_args[-2:] + p_out * in_norm

                p_c, p_args, p_out = c, args, out_vec

            for i in reversed(to_remove):
                self.commands.pop(i)
                repeat = True
