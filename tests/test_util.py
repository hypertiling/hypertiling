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


if __name__ == "__main__":
    with PrintTest() as stream:
        print("test_output")
        print(123)
        values = stream.get()

    print("Outside now:")
    print(values)
    raise ValueError("test")
