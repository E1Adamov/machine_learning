import os

from root import ROOT_PATH


def get_path_in_repo(*rel_path_parts: str) -> str:
    return os.path.join(ROOT_PATH, *rel_path_parts)


def df_cell_red_color(value):
    color = "\033[91m" if value < 90 else ""  # ANSI escape code for red text
    return f"{color}{value}\033[0m"  # Reset color after the value
