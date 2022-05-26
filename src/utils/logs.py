import os
import time


def get_log_filename(model_name: str):
    logdir = os.path.join(os.curdir, "logs")
    log_name = time.strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S")
    return os.path.join(logdir, log_name)
