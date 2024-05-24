import sys
import io
from unittest import TestCase
from unittest.mock import patch
from hypertiling.ion import htprint, set_verbosity_level

class TestVerbosity(TestCase):

    def test_htprint_lower_verbosity(self):
        set_verbosity_level("Warning")

        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            htprint("Debug", "This is a debug message")
            self.assertEqual(fake_out.getvalue(), "")


        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            htprint("Warning", "This is a warning message")
            self.assertEqual(fake_out.getvalue(), "[hypertiling] Warning: This is a warning message\n")



    def test_htprint_greater_verbosity(self):
        set_verbosity_level("Debug")

        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            htprint("Debug", "This is a debug message")
            self.assertEqual(fake_out.getvalue(), "[hypertiling] Debug: This is a debug message\n")


        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            htprint("Warning", "This is a warning message")
            self.assertEqual(fake_out.getvalue(), "[hypertiling] Warning: This is a warning message\n")