"""Rich rendering helpers for analyst reports.

The agents write Markdown, but many reports embed dense pipe tables. This module
splits those tables out so Dash can render them as fixed, readable tables and
add a compact chart when the table contains numeric series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dash import dash_table, dcc, html


@dataclass
class MarkdownBlock:
    content: str


@dataclass
class TableBlock:
    headers: list[str]
    rows: list[list[str]]
    title: Optional[str] = None


def _is_table_row(line: str) -> bool:
    return bool(line and line.count("|") >= 2)


def _is_separator_row(line: str) -> bool:
    stripped = (line or "").replace("|", "").replace(" ", "")
    return bool(stripped) and all(ch in "-:" for ch in stripped)


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _unique_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique = []
    for index, header in enumerate(headers):
        clean = header.strip() or f"Column {index + 1}"
        count = counts.get(clean, 0)
        counts[clean] = count + 1
        unique.append(clean if count == 0 else f"{clean} {count + 1}")
    return unique


def parse_table_block(lines: list[str], title: Optional[str] = None) -> Optional[TableBlock]:
    rows = [_split_row(line) for line in lines if _is_table_row(line)]
    rows = [row for row in rows if row and not _is_separator_row("|".join(row))]
    if len(rows) < 2:
        return None

    width = max(len(row) for row in rows)
    normalized = [(row + [""] * width)[:width] for row in rows]
    headers = _unique_headers(normalized[0])
    body = normalized[1:]
    if not body:
        return None
    return TableBlock(headers=headers, rows=body, title=title)


def split_report_blocks(content: str) -> list[MarkdownBlock | TableBlock]:
    """Split Markdown content into normal Markdown and parsed pipe-table blocks."""
    blocks: list[MarkdownBlock | TableBlock] = []
    markdown_lines: list[str] = []
    lines = (content or "").splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        if _is_table_row(line):
            title = None
            if markdown_lines and markdown_lines[-1].strip() and not markdown_lines[-1].lstrip().startswith(("#", "-", "*")):
                possible_title = markdown_lines[-1].strip()
                if len(possible_title) <= 80 and "table" in possible_title.lower():
                    title = possible_title
                    markdown_lines.pop()

            if markdown_lines:
                blocks.append(MarkdownBlock("\n".join(markdown_lines).strip()))
                markdown_lines = []

            table_lines = []
            while i < len(lines) and _is_table_row(lines[i]):
                table_lines.append(lines[i])
                i += 1
            table = parse_table_block(table_lines, title=title)
            if table:
                blocks.append(table)
            else:
                blocks.append(MarkdownBlock("\n".join(table_lines).strip()))
            continue

        markdown_lines.append(line)
        i += 1

    if markdown_lines:
        blocks.append(MarkdownBlock("\n".join(markdown_lines).strip()))

    return [block for block in blocks if not isinstance(block, MarkdownBlock) or block.content]


def table_to_chart_figure(table: TableBlock):
    """Charts are intentionally disabled for analyst reports.

    Analyst tables often mix prices, percentages, ratings, and text labels in a
    way that produces misleading auto-inferred charts. Keep this stub so older
    imports fail gracefully while report rendering stays table-only.
    """
    return None


def _markdown_component(content: str, min_height: Optional[str] = None):
    style = {
        "color": "#E2E8F0",
        "line-height": "1.6",
    }
    if min_height:
        style["min-height"] = min_height
    return dcc.Markdown(
        content,
        mathjax=True,
        highlight_config={"theme": "dark"},
        dangerously_allow_html=False,
        className="enhanced-markdown-content report-rich-markdown",
        style=style,
    )


def _table_component(table: TableBlock):
    columns = [{"name": header, "id": f"col_{i}"} for i, header in enumerate(table.headers)]
    data = [
        {f"col_{i}": row[i] if i < len(row) else "" for i in range(len(table.headers))}
        for row in table.rows
    ]
    style_data_conditional = [
        {"if": {"row_index": "odd"}, "backgroundColor": "rgba(15, 23, 42, 0.55)"},
    ]
    for col in columns:
        col_id = col["id"]
        style_data_conditional.extend(
            [
                {"if": {"filter_query": f'{{{col_id}}} contains "BUY"', "column_id": col_id}, "color": "#34D399", "fontWeight": "700"},
                {"if": {"filter_query": f'{{{col_id}}} contains "LONG"', "column_id": col_id}, "color": "#34D399", "fontWeight": "700"},
                {"if": {"filter_query": f'{{{col_id}}} contains "SELL"', "column_id": col_id}, "color": "#F87171", "fontWeight": "700"},
                {"if": {"filter_query": f'{{{col_id}}} contains "SHORT"', "column_id": col_id}, "color": "#F87171", "fontWeight": "700"},
                {"if": {"filter_query": f'{{{col_id}}} contains "HOLD"', "column_id": col_id}, "color": "#FBBF24", "fontWeight": "700"},
                {"if": {"filter_query": f'{{{col_id}}} contains "NEUTRAL"', "column_id": col_id}, "color": "#FBBF24", "fontWeight": "700"},
            ]
        )

    children = []
    if table.title:
        children.append(html.Div(table.title, className="report-table-title"))
    children.append(
        dash_table.DataTable(
            columns=columns,
            data=data,
            page_action="native" if len(data) > 14 else "none",
            page_size=14,
            sort_action="native",
            fill_width=True,
            style_as_list_view=True,
            style_table={"overflowX": "auto", "minWidth": "100%"},
            style_header={
                "backgroundColor": "#111827",
                "color": "#F8FAFC",
                "fontWeight": "700",
                "border": "0",
                "borderBottom": "1px solid rgba(148, 163, 184, 0.28)",
                "padding": "12px 14px",
            },
            style_cell={
                "backgroundColor": "rgba(30, 41, 59, 0.66)",
                "color": "#E5E7EB",
                "border": "0",
                "borderBottom": "1px solid rgba(51, 65, 85, 0.55)",
                "fontFamily": "Inter, Segoe UI, sans-serif",
                "fontSize": "13px",
                "lineHeight": "1.45",
                "padding": "11px 14px",
                "textAlign": "left",
                "whiteSpace": "normal",
                "height": "auto",
                "minWidth": "130px",
                "maxWidth": "320px",
            },
            style_data_conditional=style_data_conditional,
        )
    )
    return html.Div(children, className="report-table-block")


def create_rich_report_content(content: str, min_height: str = "1000px"):
    blocks = split_report_blocks(content)
    table_count = sum(isinstance(block, TableBlock) for block in blocks)
    if not table_count:
        return _markdown_component(content, min_height=min_height)

    children = []
    for block in blocks:
        if isinstance(block, MarkdownBlock):
            children.append(_markdown_component(block.content))
            continue

        children.append(_table_component(block))

    return html.Div(children, className="report-renderer", style={"minHeight": min_height})
