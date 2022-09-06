from typing import Iterable, Optional
import io
import sys


class PrintTest:

    """
    Helper class for the unit tests.
    Can be used to evaluate print-statements
    """

    def __init__(self):
        self.__std_out = sys.stdout
        self._capturedOutput = io.StringIO()

    def __enter__(self):
        sys.stdout = self._capturedOutput
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.__std_out

    def get(self):
        """
        Returns all printed strings
        return: str = string of all prints
        """
        return self._capturedOutput.getvalue()


class Progress:

    def __init__(self, iter: Iterable, length: Optional[int] = None, width: int = 100):
        self.iter = iter
        self.length = length if length is not None else len(iter)
        self.width = width

    def __iter__(self):
        for i, e in enumerate(self.iter):
            ratio = i / self.length
            ratio_ = int(self.width * ratio)
            print("\r|" + "=" * ratio_ + " " * (self.width - ratio_) + f"| {ratio:.2%}", end="", file=sys.__stdout__)
            yield e
        print("\r|" + "=" * self.width + "| 100%", end="", file=sys.__stdout__)


if __name__ == "__main__":
    # test printTest()
    with PrintTest() as stream:
        print("test_output")
        print(123)
        values = stream.get()

    print("Outside now:")
    print(values)

    # test Progress
    import time
    n = 100
    for a in Progress(range(n), n):
        time.sleep(0.1)
