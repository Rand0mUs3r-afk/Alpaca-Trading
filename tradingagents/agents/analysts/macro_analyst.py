from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
import json
from tradingagents.prompts import load_prompt, render_prompt

try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def _macro_calls(message):
    if getattr(message, "invalid_tool_calls", None):
        raise ValueError("MACRO_INVALID_TOOL_CALL")
    calls = getattr(message, "tool_calls", None)
    if not calls:
        calls = (getattr(message, "additional_kwargs", {}) or {}).get("tool_calls") or []
    if not isinstance(calls, (list, tuple)):
        raise ValueError("MACRO_INVALID_TOOL_CALL_LIST")
    parsed, ids = [], set()
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("MACRO_INVALID_TOOL_CALL")
        function = call.get("function") or {}
        if not isinstance(function, dict):
            raise ValueError("MACRO_INVALID_TOOL_CALL")
        name, call_id = call.get("name") or function.get("name"), call.get("id")
        if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
            raise ValueError("MACRO_INVALID_TOOL_NAME_OR_ID")
        if call_id in ids:
            raise ValueError("MACRO_DUPLICATE_TOOL_CALL_ID")
        ids.add(call_id)
        args = call["args"] if "args" in call else function.get("arguments", {})
        try:
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict):
                raise ValueError()
            args = json.loads(json.dumps(args, allow_nan=False))
        except (ValueError, TypeError):
            raise ValueError("MACRO_INVALID_TOOL_ARGUMENTS") from None
        parsed.append({"name": name, "id": call_id, "args": args})
    return parsed


def _macro_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content if isinstance(block, dict)
                         and block.get("type") in ("text", "output_text")
                         and isinstance(block.get("text"), str)).strip()
    return ""


def create_macro_analyst(llm, toolkit):
    def analyze(state):
        current_date = state["trade_date"]
        ticker = state.get("company_of_interest", "MARKET")
        tools, sources = [], []
        if toolkit.has_fred():
            tools.extend([toolkit.get_macro_analysis, toolkit.get_economic_indicators,
                          toolkit.get_yield_curve_analysis])
            sources.append("FRED macro data")
        if toolkit.has_fxmacrodata():
            tools.extend([toolkit.get_global_macro_indicators, toolkit.get_economic_release_calendar])
            if toolkit.has_fxmacrodata_key():
                tools.append(toolkit.get_fx_rates)
            sources.append("FXMacroData global macro releases and calendar")
        news_available = toolkit.config["online_tools"] and toolkit.has_openai_web_search()
        if news_available:
            tools.append(toolkit.get_macro_news_openai)
            sources.append("OpenAI macro web search")
        if not tools:
            report = "UNAVAILABLE: No external macro source is configured; no macro conditions measured."
            return {"messages": list(state["messages"]) + [AIMessage(content=report)], "macro_report": report}
        guidance = (
            " Combine all currently available macro tools before concluding."
            f" Active sources: {', '.join(sources)}."
            + (f" Use `get_macro_news_openai(curr_date, ticker_context='{ticker}')` for relevance." if news_available else "")
        )
        system_message = render_prompt("analysts/macro_system", source_guidance=guidance)
        asset_context = (
            f"Asset context is {ticker}. Focus on macroeconomic conditions that affect overall "
            "market sentiment and sector rotation. Use only returned source evidence. "
            "If sources are unavailable, state that explicitly; do not invent current macro conditions."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("shared/analyst_tool_system")),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(system_message=system_message, tool_names=", ".join(tool.name for tool in tools),
                   current_date=current_date, ticker=ticker, asset_context=asset_context)
        history = list(state["messages"])
        try:
            formatted = prompt.format_messages(messages=history)
            capture_agent_prompt("macro_report", formatted[0].content if formatted else system_message, ticker)
        except Exception as exc:
            print(f"[MACRO] Prompt capture unavailable: {type(exc).__name__}")
        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(history)
        calls = _macro_calls(result)
        tool_map = {tool.name: tool for tool in tools}
        cache, counts, used_ids = {}, {}, set()
        limit = int(toolkit.config.get("max_tool_iterations_per_agent", 8))
        repeat_limit = int(toolkit.config.get("max_same_tool_call_repeats", 1))
        iterations = 0
        while calls and iterations < limit:
            for call in calls:
                if call["name"] not in tool_map:
                    raise ValueError("MACRO_UNKNOWN_TOOL")
                if call["id"] in used_ids:
                    raise ValueError("MACRO_REUSED_TOOL_CALL_ID")
            used_ids.update(call["id"] for call in calls)
            iterations += 1
            history.append(result)
            for call in calls:
                name, args = call["name"], call["args"]
                signature = f"{name}:{json.dumps(args, sort_keys=True, allow_nan=False)}"
                counts[signature] = counts.get(signature, 0) + 1
                if signature in cache:
                    output = (
                        f"Duplicate request. Full evidence for identical tool name and arguments "
                        f"is in earlier ToolMessage tool_call_id={json.dumps(cache[signature])}. "
                        "Reuse the original evidence; do not request it again."
                    )
                elif counts[signature] > repeat_limit:
                    output = f"Skipped repeated tool call '{name}'. No new evidence was collected."
                else:
                    # Tool and provider exceptions propagate unchanged to the runner.
                    tool = tool_map[name]
                    output = tool.invoke(args) if hasattr(tool, "invoke") else tool.run(**args)
                    cache[signature] = call["id"]
                history.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
            result = chain.invoke(history)
            calls = _macro_calls(result)
        if calls:
            raise ValueError("MACRO_TOOL_ITERATION_LIMIT")
        report = _macro_text(result.content)
        if not report:
            raise ValueError("MACRO_EMPTY_REPORT")
        if "FINAL TRANSACTION PROPOSAL:" not in report:
            final_prompt = render_prompt("analysts/macro_final_recommendation",
                                         current_date=current_date, analysis_content=report)
            final = llm.invoke(final_prompt)
            final_text = _macro_text(final.content) if hasattr(final, "content") else str(final).strip()
            if not final_text:
                raise ValueError("MACRO_EMPTY_FINAL_RECOMMENDATION")
            report += "\n\n---\n\n## Final Recommendation\n\n" + final_text
        result = AIMessage(content=report)
        history.append(result)
        return {"messages": history, "macro_report": report}

    def macro_analyst_node(state):
        try:
            return analyze(state)
        except Exception as exc:
            print(f"[MACRO] Analysis failed: {type(exc).__name__}")
            raise
    return macro_analyst_node
