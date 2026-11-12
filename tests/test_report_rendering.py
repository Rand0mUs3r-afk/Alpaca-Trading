import unittest

from webui.utils.report_rendering import (
    TableBlock,
    split_report_blocks,
    table_to_chart_figure,
)


class ReportRenderingTests(unittest.TestCase):
    def test_splits_markdown_tables_out_of_report_text(self):
        content = """
## Market Setup
Momentum improved into the close.

Summary Table
| Level | Price | Confidence |
| --- | ---: | --- |
| Support | 182.50 | 0.70 |
| Target | 195.00 | 0.64 |

Final note stays markdown.
"""

        blocks = split_report_blocks(content)

        self.assertEqual(len(blocks), 3)
        self.assertIsInstance(blocks[1], TableBlock)
        self.assertEqual(blocks[1].title, "Summary Table")
        self.assertEqual(blocks[1].headers, ["Level", "Price", "Confidence"])
        self.assertEqual(blocks[1].rows[0][0], "Support")

    def test_numeric_tables_do_not_create_charts(self):
        table = TableBlock(
            headers=["Setup", "Price", "Probability"],
            rows=[
                ["Support", "100.50", "0.70"],
                ["Breakout", "101.75", "0.62"],
                ["Target", "103.00", "0.58"],
            ],
        )

        self.assertIsNone(table_to_chart_figure(table))

    def test_date_only_tables_do_not_create_misleading_chart(self):
        table = TableBlock(
            headers=["Event", "Date", "Impact"],
            rows=[
                ["CPI release", "2026-05-13", "High"],
                ["Earnings revision", "2026-05-16", "Medium"],
            ],
        )

        self.assertIsNone(table_to_chart_figure(table))


if __name__ == "__main__":
    unittest.main()
