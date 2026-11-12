"""Our own modules must not emit Pydantic deprecation warnings on import.

Third-party warnings are out of scope; this guards the code we own so it
keeps working when Pydantic 3 removes the deprecated class-based config.
"""

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class OwnedDeprecationWarningTests(unittest.TestCase):
    def test_gpt5_llm_import_emits_no_pydantic_deprecation(self):
        code = (
            "import warnings\n"
            "from pydantic import PydanticDeprecatedSince20\n"
            "with warnings.catch_warnings():\n"
            "    warnings.simplefilter('error', PydanticDeprecatedSince20)\n"
            "    import tradingagents.agents.utils.gpt5_llm\n"
            "print('CLEAN')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300,
        )
        self.assertIn(
            "CLEAN",
            result.stdout,
            f"importing gpt5_llm raised a Pydantic deprecation:\n{result.stderr[-2000:]}",
        )


if __name__ == "__main__":
    unittest.main()
