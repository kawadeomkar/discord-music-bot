import functools
import logging
import sys


def debug(func):
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        sig = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
        print(f"Calling {func.__name__}({sig})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


def validate_input(*expected_args):
    def validate_outer(func):
        @functools.wraps(func)
        def validate_wrap(*args, **kwargs):
            for exp in expected_args:
                if exp not in args or exp not in kwargs:
                    raise Exception(f"Expected argument does not exist {exp}")
            return func(*args, **kwargs)

        return validate_wrap

    return validate_outer


def setLogger(name: str) -> logging:
    root = logging.getLogger(name)
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root
