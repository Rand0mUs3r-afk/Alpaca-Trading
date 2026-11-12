"""
Report validation utility to ensure UI shows complete, final reports
"""

import re
from typing import Dict, Optional


def is_report_complete(report_content: str, report_type: str) -> bool:
    """
    Check if a report appears to be complete based on content analysis
    
    Args:
        report_content: The report text to validate
        report_type: Type of report (market_report, sentiment_report, etc.)
    
    Returns:
        bool: True if report appears complete, False otherwise
    """
    if not report_content or len(report_content.strip()) < 100:
        return False
    
    # Check for common completion indicators
    completion_indicators = [
        "## Summary",
        "## Conclusion", 
        "## Trading Implications",
        "## Recommendation",
        "| Key Metric |",  # Markdown table
        "| Metric |",
        "**Recommendation:**",
        "## Key Points",
        "### Trading Implications"
    ]
    
    # Check if report has at least one completion indicator
    has_completion_indicator = any(indicator.lower() in report_content.lower() 
                                 for indicator in completion_indicators)
    
    # Check for markdown table (common in complete reports)
    has_table = bool(re.search(r'\|.*\|.*\|', report_content))
    
    # Check minimum length by report type
    min_lengths = {
        "market_report": 1000,
        "sentiment_report": 1000,
        "news_report": 1000,
        "fundamentals_report": 1000,
        "macro_report": 1000,
    }
    
    min_length = min_lengths.get(report_type, 300)
    meets_length = len(report_content) >= min_length
    
    # Report is complete if it has indicators AND meets length OR has table structure
    return (has_completion_indicator and meets_length) or has_table


def validate_reports_for_ui(reports: Dict[str, Optional[str]]) -> Dict[str, str]:
    """
    Validate and potentially mark incomplete reports for UI display
    
    Args:
        reports: Dictionary of report_type -> report_content
        
    Returns:
        Dictionary with validated reports, incomplete ones marked as "In Progress..."
    """
    validated_reports = {}
    
    for report_type, content in reports.items():
        if not content:
            validated_reports[report_type] = f"No {report_type.replace('_', ' ').title()} available yet."
            continue
            
        if is_report_complete(content, report_type):
            validated_reports[report_type] = content
        else:
            # Mark as in progress with partial content
            validated_reports[report_type] = f"""
## 🔄 {report_type.replace('_', ' ').title()} - In Progress

**Status:** Analysis currently running...

**Partial Content Preview:**
{content[:200]}{'...' if len(content) > 200 else ''}

---
*Full report will appear here when analysis completes*
"""
    
    return validated_reports


def get_report_completion_status(reports: Dict[str, Optional[str]]) -> Dict[str, str]:
    """
    Get completion status for each report type
    
    Returns:
        Dict mapping report_type to status: "complete", "incomplete", or "missing"
    """
    status = {}
    
    for report_type, content in reports.items():
        if not content:
            status[report_type] = "missing"
        elif is_report_complete(content, report_type):
            status[report_type] = "complete"
        else:
            status[report_type] = "incomplete"
    
    return status 