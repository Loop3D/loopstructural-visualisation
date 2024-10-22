from LoopStructural.utils import rng
from matplotlib.colors import to_hex


def random_colour():
    return to_hex(rng.random(3))
