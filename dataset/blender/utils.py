import struct
import random
from math import radians


def deg2rad(degs):
    if type(degs) == tuple or type(degs) == list:
        return tuple(radians(t) for t in degs)
    return radians(degs)


def hex2rgb(hex):
    int_tuple = struct.unpack('BBB', bytes.fromhex(hex))
    return tuple([val / 255 for val in int_tuple])


def random_like_color(grayscale=False, lower_limit=0.0, upper_limit=1.0):
    if grayscale:
        r = random.uniform(lower_limit, upper_limit)
        rr = random.gauss(r, 0.05)
        g = random.gauss(r, 0.05)
        b = random.gauss(r, 0.05)
        return (rr, g, b)

    r = random.uniform(lower_limit, upper_limit)
    g = random.uniform(lower_limit, upper_limit)
    b = random.uniform(lower_limit, upper_limit)

    return (r, g, b)
