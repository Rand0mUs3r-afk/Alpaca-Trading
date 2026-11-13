"""Run the embedded Groq diagnostic regressions in standard test discovery.
No API keys, network requests or order submissions are used.
"""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from scripts import free_schema_compat as compat


class GroqDiagnosticRegressionTests(unittest.TestCase):
    def test_embedded_diagnostic_suite(self):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            passed = compat.self_test()
        self.assertIs(passed, True, output.getvalue())


if __name__ == '__main__':
    unittest.main()
