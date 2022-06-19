import argparse


def get_script_args():
    """Returns the arguments passed to the script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        help="train or use stored model",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--plot",
        help="plot graphs",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()
    return args
