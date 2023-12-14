import os

import requests

from helpers.protocols import HasLtMethod
from root import ROOT_PATH


def get_path_in_repo(*rel_path_parts: str) -> str:
    return os.path.join(ROOT_PATH, *rel_path_parts)


def num_to_str_red_color(num: HasLtMethod, threshold: int = 90) -> str:
    """
    casts the <value> to string and colors red if it's below <threshold>
    """
    color = "\033[91m" if num < threshold else ""  # ANSI escape code for red text
    return f"{color}{num}\033[0m"  # Reset color after the value


def download_large_file(url, save_path, chunk_size=8192):
    with requests.get(url, stream=True) as response:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
