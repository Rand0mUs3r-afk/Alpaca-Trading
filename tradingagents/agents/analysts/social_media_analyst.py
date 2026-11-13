from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
import json
from tradingagents.prompts import load_prompt, render_prompt

try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def _social_tool_calls(message):
    """Read canonical or legacy calls without altering the original AI message."""
    if getattr(message, "invalid_tool_calls", None):
        raise ValueError("SOCIAL_INVALID_TOOL_CALL")
    calls = getattr(message, "tool_calls", None)
    if not calls:
        calls = (getattr(message, "additional_kwargs", {}) or {}).get("tool_calls", [])
    if calls is None:
        calls = []
    if not isinstance(calls, (list, tuple)):
        raise ValueError("SOCIAL_INVALID_TOOL_CALL_LIST")
    parsed, ids = [], set()
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("SOCIAL_INVALID_TOOL_CALL")
        function = call.get("function") or {}
        if not isinstance(function, dict):
            raise ValueError("SOCIAL_INVALID_TOOL_CALL")
        name = call.get("name") or function.get("name")
        call_id = call.get("id")
        args = call["args"] if "args" in call else function.get("arguments", {})
        if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
            raise ValueError("SOCIAL_INVALID_TOOL_NAME_OR_ID")
        if call_id in ids:
            raise ValueError("SOCIAL_DUPLICATE_TOOL_CALL_ID")
        ids.add(call_id)
        try:
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError()
            args = json.loads(json.dumps(args, allow_nan=False))
        except (TypeError, ValueError):
            raise ValueError("SOCIAL_INVALID_TOOL_ARGUMENTS") from None
        parsed.append({"name": name, "id": call_id, "args": args})
    return parsed


def _social_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content if isinstance(block, dict)
                         and block.get("type") in ("text", "output_text")
                         and isinstance(block.get("text"), str)).strip()
    return ""


def create_social_media_analyst(llm, toolkit):
    def social_media_analyst_node(state):
        enabled = toolkit.config.get("social_llm_enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("SOCIAL_INVALID_LLM_SETTING")
        if not enabled:
            # Explicitly unavailable, not neutral or bullish. No tools, model
            # calls, fabricated metrics, or final trade proposal in this path.
            report = "UNAVAILABLE: Social sentiment not measured in free profile."
            print("[SOCIAL] Skipped: social_llm_enabled=False; no sentiment measured.", flush=True)
            return {"messages": [AIMessage(content=report)], "sentiment_report": report}
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        is_crypto = "/" in ticker or "USD" in ticker.upper() or "USDT" in ticker.upper()
        openai_available = toolkit.has_openai_web_search()
        reddit_tool = toolkit.get_reddit_news if is_crypto else toolkit.get_reddit_stock_info
        tools = [reddit_tool]
        if toolkit.config["online_tools"] and openai_available:
            tools.insert(0, toolkit.get_stock_news_openai)
        source_labels = ["Reddit"]
        if toolkit.config["online_tools"] and openai_available:
            source_labels.insert(0, "OpenAI web-search sentiment")
        source_guidance = (
            " Use all currently available social tools before concluding."
            f" Active sources: {', '.join(source_labels)}."
            + (" Use `get_reddit_news(curr_date)` for crypto context." if is_crypto else " Use `get_reddit_stock_info(ticker, curr_date)` for stock context.")
        )
        system_message = render_prompt("analysts/social_system", source_guidance=source_guidance)
        asset_context = f"The current company we want to analyze is {ticker}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("shared/analyst_tool_system")),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join(tool.name for tool in tools))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)
        prompt = prompt.partial(asset_context=asset_context)
        try:
            formatted = prompt.format_messages(messages=list(state["messages"]))
            complete_prompt = formatted[0].content if formatted and hasattr(formatted[0], "content") else render_prompt(
                "shared/analyst_tool_system", tool_names=", ".join(tool.name for tool in tools),
                system_message=system_message, current_date=current_date, asset_context=asset_context)
            capture_agent_prompt("sentiment_report", complete_prompt, ticker)
        except Exception as exc:
            print(f"[SOCIAL] Prompt capture unavailable: {type(exc).__name__}")
            capture_agent_prompt("sentiment_report", system_message, ticker)

        chain = prompt | (llm.bind_tools(tools) if tools else llm)
        messages_history = list(state["messages"])
        result = chain.invoke(messages_history)
        max_tool_iterations = int(toolkit.config.get("max_tool_iterations_per_agent", 8))
        max_same_call_repeats = int(toolkit.config.get("max_same_tool_call_repeats", 1))
        tool_call_counts, tool_result_cache = {}, {}
        iteration_count = 0
        calls = _social_tool_calls(result)
        tool_map = {tool.name: tool for tool in tools}
        used_ids = set()
        while tools and calls and iteration_count < max_tool_iterations:
            # Validate the whole batch before executing any tool from that batch.
            for call in calls:
                if call["name"] not in tool_map:
                    raise ValueError("SOCIAL_UNKNOWN_TOOL")
                if call["id"] in used_ids:
                    raise ValueError("SOCIAL_REUSED_TOOL_CALL_ID")
            used_ids.update(call["id"] for call in calls)
            iteration_count += 1
            # Preserve provider metadata and keep a multi-call reply as one message.
            messages_history.append(result)
            for call in calls:
                tool_name, tool_args = call["name"], call["args"]
                tool_fn = tool_map[tool_name]
                signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, allow_nan=False)}"
                count = tool_call_counts.get(signature, 0) + 1
                tool_call_counts[signature] = count
                if signature in tool_result_cache:
                    tool_result = tool_result_cache[signature]
                elif count > max_same_call_repeats:
                    tool_result = (
                        f"Skipped repeated tool call '{tool_name}' with identical arguments "
                        f"after {max_same_call_repeats} execution(s). Reuse previous evidence."
                    )
                else:
                    try:
                        tool_result = tool_fn.invoke(tool_args) if hasattr(tool_fn, "invoke") else tool_fn.run(**tool_args)
                        tool_result_cache[signature] = str(tool_result)
                    except Exception as tool_err:
                        # Do not echo credentials or full provider errors into the LLM.
                        tool_result = f"Error running tool '{tool_name}': {type(tool_err).__name__}. No usable result from this call."
                messages_history.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
            result = chain.invoke(messages_history)
            calls = _social_tool_calls(result)

        if tools and calls:
            result = AIMessage(content=(_social_text(result.content)
                + f"\n\nTool-loop halted after {max_tool_iterations} iterations to prevent endless retries.").strip())
        analysis_content = _social_text(result.content)
        if not analysis_content:
            analysis_content = (
                "The analyst did not return a full social narrative. "
                "Be explicit about that limitation in the final recommendation."
            )
        if "FINAL TRANSACTION PROPOSAL:" not in analysis_content:
            final_prompt = render_prompt("analysts/social_final_recommendation", ticker=ticker,
                                         analysis_content=analysis_content)
            final_result = llm.invoke(final_prompt)
            final_content = _social_text(final_result.content) if hasattr(final_result, "content") else str(final_result)
            result = AIMessage(content=analysis_content + "\n\n---\n\n## Final Recommendation\n\n" + final_content)
        else:
            result = AIMessage(content=analysis_content)
        return {"messages": [result], "sentiment_report": result.content}
    return social_media_analyst_node


def self_test():
    """Offline disabled-path checks; no model, data-tool, or broker requests."""
    import unittest
    from types import SimpleNamespace
    from contextlib import redirect_stdout
    from io import StringIO
    class NoModelCalls:
        def __getattr__(self, name):
            raise AssertionError("Model access is forbidden in disabled social analysis")
    class Tests(unittest.TestCase):
        def node(self, setting=False):
            return create_social_media_analyst(NoModelCalls(),
                SimpleNamespace(config={"social_llm_enabled": setting}))
        def test_disabled_path_needs_no_tools_or_state_fields(self):
            with redirect_stdout(StringIO()):
                result = self.node()({})
            self.assertTrue(result["sentiment_report"].startswith("UNAVAILABLE:"))
            self.assertEqual(len(result["messages"]), 1)
        def test_report_is_short_and_not_a_trade_signal(self):
            with redirect_stdout(StringIO()):
                result = self.node()({})
            report = result["sentiment_report"]
            self.assertLess(len(report), 80)
            self.assertNotIn("FINAL TRANSACTION PROPOSAL", report)
            self.assertNotIn("BUY", report)
            self.assertNotIn("SELL", report)
            self.assertFalse(getattr(result["messages"][0], "tool_calls", []))
        def test_disabled_path_does_not_mutate_input_state(self):
            prior = AIMessage(content="Prior report")
            state = {"messages": [prior], "company_of_interest": "AAPL"}
            with redirect_stdout(StringIO()):
                self.node()(state)
            self.assertEqual(state, {"messages": [prior], "company_of_interest": "AAPL"})
        def test_malformed_setting_fails_before_model_or_tool_access(self):
            for setting in ("false", None, 0):
                with self.subTest(setting_type=type(setting).__name__):
                    with self.assertRaisesRegex(ValueError, "SOCIAL_INVALID_LLM_SETTING"):
                        self.node(setting)({})
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


if __name__ == '__main__':
    import sys
    if sys.argv[1:] != ['--self-test']:
        raise SystemExit('Only --self-test is available as a command.')
    raise SystemExit(0 if self_test() else 1)
