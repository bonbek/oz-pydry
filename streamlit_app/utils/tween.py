import time
import math

def ease_out(t):
    return 1 if t == 1 else 1 - 2**(-10 * t)

def ease_in_out(t):
    return -(math.cos(math.pi * t) - 1) / 2

def tween(duration, ease = ease_out):
    ease = ease or (lambda x: x)
    s = time.time()
    t = 0
    while True:
        yield ease(t / duration)
        e = time.time()
        time.sleep(min(.05, duration - t)) 
        t = e - s
        if t >= duration:
            break
    yield 1