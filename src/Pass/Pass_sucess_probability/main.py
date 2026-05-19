import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Pass.Pass_sucess_probability.trainer import Train
    from Pass.Pass_sucess_probability.test import Test
else:
    from .trainer import Train
    from .test import Test


if __name__ == "__main__":
    Train()
    Test()


