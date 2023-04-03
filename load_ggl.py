from pathlib import Path

from utils.dataloader import load_data, encode_data, FontsDataset
import shutil
import time
import numpy as np
from torch.utils.data import DataLoader

# load_data()
encode_data()

# dataset = FontsDataset(test=True, download=True, download_size=40)

# dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
#
# print(np.load('data/encoded/train/!the_black_bloc-bold/a.npy').dtype)
# print(dataset[0][0].dtype)
# print(next(iter(dataloader))[0].dtype)

# if __name__ == '__main__':
#     print('in main')
#     for i, d in enumerate(train_dataloader):
#         if i % 10 == 0:
#             print(i)

# with open('data/blacklist2.txt', 'r') as f:
#     blacklist = [i.strip() for i in f.readlines()]
#
# for p in Path('data/svg').iterdir():
#     if p.stem in blacklist:
#         shutil.rmtree(p)
#         print('deleted', p)
# letters = []
#
# start = time.time()
#
# for i in range(256):
#     d, ch, font = dataset[i]
#     letters.append(d)
#
# end = time.time()
#
# print("Duration", end - start)
#
# #
# # a = np.array([[1, 2], [2, 3]])
# # np.save('test_numpy.npy', a)
# # b = np.load('test_numpy.npy')
# # print(b)
