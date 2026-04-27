import sys
from .trainer import Train
from .test import Test


if __name__ == "__main__":
    # Support command-line argument to choose between train and test
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        Train()
    elif mode == "test":
        Test()
    elif mode == "train_test":
        Train()
        Test()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [train|test|train_test]")
        sys.exit(1)
