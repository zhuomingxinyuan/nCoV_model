from typing import List
import pandas as pd


def check_type(to_be_checked, expected_type):
    if type(to_be_checked) != expected_type:
        raise Exception("Pass in right type")


class Result:
    def __init__(self, name, cols: List[str]):
        self.name = name
        self.cols = cols
        self._d = None

    def init_df(self):
        self._d = pd.DataFrame({}, columns=self.cols, index=[])
        return self

    def add_row(self, row: List, idx: int):
        if len(row) != len(self.cols):
            raise Exception("Wrong row")
        self._d.loc[idx] = row

    def get_df(self):
        return self._d


def print_with_title(title, body=None):
    print("")
    print("-" * 10 + title.upper() + "-" * 10)
    print(body)
    if body is not None:
        print("-" * 10 + "-" * len(title) + "-" * 10)


# print_with_title(title="title", body="this is body")
