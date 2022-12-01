import os
current_filepath = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))
ROOT_PATH = os.path.abspath(os.path.join(SRC_PATH, os.pardir))
DATA_PATH = os.path.join(ROOT_PATH, "dataset")
LOG_PATH = os.path.join(ROOT_PATH, "log")