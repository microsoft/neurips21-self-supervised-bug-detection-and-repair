from html.parser import HTMLParser
from typing import Set

import requests


class SimpleFormatParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.__in_a_tag = False
        self.all_packages = set()

    def handle_starttag(self, tag, attrs):
        self.__in_a_tag = tag == "a"

    def handle_data(self, data):
        if self.__in_a_tag:
            self.all_packages.add(data)


def get_all_pypi_packages() -> Set[str]:
    parser = SimpleFormatParser()
    data = requests.get("https://pypi.org/simple/")
    parser.feed(data.text)
    return parser.all_packages
