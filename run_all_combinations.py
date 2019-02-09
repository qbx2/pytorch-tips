import itertools
import os
import sys


def bind_key(item):
    key, values = item

    for value in values:
        yield key, value


prefix = ' '.join(sys.argv[1:])
combinations = {
    '--cuda': [None],
    '--dim': [256, 512, 1024],
    '--dropout': [0, 0.05],
    '--epoch': [500],
    '--learning-rate': [1, 0.1, 0.01],
    '--num-layers': [2, 4, 8, 16, 32, 64, 128],
}

for combination in itertools.product(*map(bind_key, combinations.items())):
    args = [prefix]

    for option in combination:
        args += map(str, filter(lambda p: p is not None, option))

    cmd = ' '.join(args)
    print('$ ' + cmd)
    os.system(cmd)
