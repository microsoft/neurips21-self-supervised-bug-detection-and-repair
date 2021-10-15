import unittest

from libcst.metadata import CodeRange

from buglab.utils.cstutils import relative_range
from buglab.utils.text import get_text_in_range


class TestCodeRangeUtils(unittest.TestCase):
    def test_relative_range(self):
        #  Load some text file (this one, for now)
        with open(__file__) as f:
            text = f.read()

        self.__assert_relative_and_absolute_are_equal(
            absolute_range=CodeRange((19, 4), (26, 50)), base_range=CodeRange((8, 0), (27, 0)), text=text
        )

        self.__assert_relative_and_absolute_are_equal(
            absolute_range=CodeRange((20, 40), (20, 50)), base_range=CodeRange((8, 0), (27, 0)), text=text
        )

        self.__assert_relative_and_absolute_are_equal(
            absolute_range=CodeRange((20, 40), (20, 50)), base_range=CodeRange((20, 0), (20, 60)), text=text
        )

        self.__assert_relative_and_absolute_are_equal(
            absolute_range=CodeRange((20, 40), (20, 50)), base_range=CodeRange((20, 20), (20, 60)), text=text
        )

        self.__assert_relative_and_absolute_are_equal(
            absolute_range=CodeRange((20, 40), (20, 50)), base_range=CodeRange((10, 10), (20, 60)), text=text
        )

    def __assert_relative_and_absolute_are_equal(
        self, absolute_range: CodeRange, base_range: CodeRange, text: str
    ) -> None:
        rel_range = relative_range(base_range, absolute_range)

        self.assertEqual(
            get_text_in_range(get_text_in_range(text, base_range), rel_range),
            get_text_in_range(text, absolute_range),
            f"Relative {rel_range} and absolute {absolute_range} ranges do not retrieve the same snippet of text.",
        )


if __name__ == "__main__":
    unittest.main()
