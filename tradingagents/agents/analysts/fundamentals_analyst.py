from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
import json
from tradingagents.prompts import load_prompt, render_prompt

try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def _fundamentals_tool_calls(message):
    """Validate canonical or legacy calls while preserving the original AI reply."""
    if getattr(message, "invalid_tool_calls", None):
        raise ValueError("FUNDAMENTALS_INVALID_TOOL_CALL")
    calls = getattr(message, "tool_calls", None)
    if not calls:
        calls = (getattr(message, "additional_kwargs", {}) or {}).get("tool_calls", [])
    if calls is None:
        calls = []
    if not isinstance(calls, (list, tuple)):
        raise ValueError("FUNDAMENTALS_INVALID_TOOL_CALL_LIST")
    parsed, ids = [], set()
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("FUNDAMENTALS_INVALID_TOOL_CALL")
        function = call.get("function") or {}
        if not isinstance(function, dict):
            raise ValueError("FUNDAMENTALS_INVALID_TOOL_CALL")
        name = call.get("name") or function.get("name")
        call_id = call.get("id")
        args = call["args"] if "args" in call else function.get("arguments", {})
        if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
            raise ValueError("FUNDAMENTALS_INVALID_TOOL_NAME_OR_ID")
        if call_id in ids:
            raise ValueError("FUNDAMENTALS_DUPLICATE_TOOL_CALL_ID")
        ids.add(call_id)
        try:
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError()
            args = json.loads(json.dumps(args, allow_nan=False))
        except (TypeError, ValueError):
            raise ValueError("FUNDAMENTALS_INVALID_TOOL_ARGUMENTS") from None
        parsed.append({"name": name, "id": call_id, "args": args})
    return parsed


def _fundamentals_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content if isinstance(block, dict)
                         and block.get("type") in ("text", "output_text")
                         and isinstance(block.get("text"), str)).strip()
    return ""


def create_fundamentals_analyst(llm, toolkit):
    def analyze(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        is_crypto = "/" in ticker or "USD" in ticker.upper() or "USDT" in ticker.upper()
        display_ticker = ticker
        if is_crypto:
            if "/" in ticker:
                display_ticker = ticker.split("/")[0]
            elif "USDT" in ticker.upper():
                display_ticker = ticker.upper().replace("USDT", "")
            elif "USD" in ticker.upper():
                display_ticker = ticker.upper().replace("USD", "")
        openai_available = toolkit.has_openai_web_search()
        finnhub_available = toolkit.has_finnhub()
        simfin_available = toolkit.has_simfin_data()
        base_openai_tools = [toolkit.get_fundamentals_openai] if toolkit.config["online_tools"] and openai_available else []
        if is_crypto:
            tools = [toolkit.get_defillama_fundamentals] + base_openai_tools
            active_sources = ["DeFiLlama"] + (["OpenAI web search"] if base_openai_tools else [])
        else:
            tools = list(base_openai_tools)
            if finnhub_available:
                tools.extend([toolkit.get_finnhub_company_insider_sentiment,
                              toolkit.get_finnhub_company_insider_transactions])
            if simfin_available:
                tools.extend([toolkit.get_simfin_balance_sheet, toolkit.get_simfin_cashflow,
                              toolkit.get_simfin_income_stmt])
            active_sources = []
            if base_openai_tools:
                active_sources.append("OpenAI web search")
            if finnhub_available:
                active_sources.append("Finnhub")
            if simfin_available:
                active_sources.append("SimFin")
        source_guidance = (
            " Use all available fundamentals tools before concluding. "
            + (f"Active sources now: {', '.join(active_sources)}." if active_sources else
               "No external fundamentals source is available; reason from existing context only.")
        )
        asset_focus = (
            "Analyze DeFi metrics like TVL changes, protocol upgrades, token unlock schedules, yield farming opportunities, and major partnership announcements that could sustain multi-day crypto price trends."
            if is_crypto else
            "Focus on earnings surprises, analyst upgrades/downgrades, insider activity, and fundamental shifts that could sustain multi-day swing moves."
        )
        system_message = render_prompt("analysts/fundamentals_system", asset_focus=asset_focus,
                                       source_guidance=source_guidance)
        asset_context = (f"The cryptocurrency we want to analyze is {display_ticker}" if is_crypto
                         else f"The company we want to look at is {ticker}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("shared/analyst_tool_system")),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join(tool.name for tool in tools))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=display_ticker if is_crypto else ticker)
        prompt = prompt.partial(asset_context=asset_context)
        try:
            formatted = prompt.format_messages(messages=list(state["messages"]))
            complete_prompt = formatted[0].content if formatted and hasattr(formatted[0], "content") else render_prompt(
                "shared/analyst_tool_system", tool_names=", ".join(tool.name for tool in tools),
                system_message=system_message, current_date=current_date, asset_context=asset_context)
            capture_agent_prompt("fundamentals_report", complete_prompt, ticker)
        except Exception as exc:
            print(f"[FUNDAMENTALS] Prompt capture unavailable: {type(exc).__name__}")
            capture_agent_prompt("fundamentals_report", system_message, ticker)

        chain = prompt | (llm.bind_tools(tools) if tools else llm)
        messages_history = list(state["messages"])
        result = chain.invoke(messages_history)
        max_tool_iterations = int(toolkit.config.get("max_tool_iterations_per_agent", 8))
        max_same_call_repeats = int(toolkit.config.get("max_same_tool_call_repeats", 1))
        tool_call_counts, tool_result_cache = {}, {}
        iteration_count = 0
        calls = _fundamentals_tool_calls(result)
        tool_map = {tool.name: tool for tool in tools}
        used_ids = set()
        if calls and not tools:
            raise ValueError("FUNDAMENTALS_UNKNOWN_TOOL")
        while calls and iteration_count < max_tool_iterations:
            for call in calls:
                if call["name"] not in tool_map:
                    raise ValueError("FUNDAMENTALS_UNKNOWN_TOOL")
                if call["id"] in used_ids:
                    raise ValueError("FUNDAMENTALS_REUSED_TOOL_CALL_ID")
            used_ids.update(call["id"] for call in calls)
            iteration_count += 1
            messages_history.append(result)
            for call in calls:
                tool_name, tool_args = call["name"], call["args"]
                tool_fn = tool_map[tool_name]
                signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, allow_nan=False)}"
                count = tool_call_counts.get(signature, 0) + 1
                tool_call_counts[signature] = count
                if signature in tool_result_cache:
                    # Keep the full original ToolMessage once; acknowledge the new ID
                    # without copying the same SEC evidence into every model request.
                    original_id = tool_result_cache[signature]
                    tool_result = (
                        f"Duplicate request. The full result for identical tool name and arguments "
                        f"is already present in ToolMessage tool_call_id={json.dumps(original_id)}. "
                        "Reuse that original evidence and finish the analysis; do not request it again."
                    )
                elif count > max_same_call_repeats:
                    tool_result = (
                        f"Skipped repeated tool call '{tool_name}' with identical arguments "
                        f"after {max_same_call_repeats} execution(s). Reuse previous evidence."
                    )
                else:
                    try:
                        tool_result = tool_fn.invoke(tool_args) if hasattr(tool_fn, "invoke") else tool_fn.run(**tool_args)
                        tool_result_cache[signature] = call["id"]
                    except Exception as tool_err:
                        tool_result = f"Error running tool '{tool_name}': {type(tool_err).__name__}. No usable result from this call."
                messages_history.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
            result = chain.invoke(messages_history)
            calls = _fundamentals_tool_calls(result)
        if calls:
            result = AIMessage(content=(_fundamentals_text(result.content)
                + f"\n\nTool-loop halted after {max_tool_iterations} iterations to prevent endless retries.").strip())
        analysis_content = _fundamentals_text(result.content)
        if not analysis_content:
            analysis_content = (
                "The analyst did not return a complete fundamentals narrative. "
                "State that limitation clearly in the final recommendation."
            )
        if "FINAL TRANSACTION PROPOSAL:" not in analysis_content:
            final_prompt = render_prompt("analysts/fundamentals_final_recommendation", ticker=ticker,
                                         analysis_content=analysis_content)
            final_result = llm.invoke(final_prompt)
            final_content = _fundamentals_text(final_result.content) if hasattr(final_result, "content") else str(final_result)
            analysis_content += "\n\n" + final_content
        result = AIMessage(content=analysis_content)
        messages_history.append(result)
        return {"messages": messages_history, "fundamentals_report": result.content}

    def fundamentals_analyst_node(state):
        try:
            return analyze(state)
        except Exception as exc:
            # Keep the original exception/response for the caller's safe diagnostics
            # and bounded provider fallback. Never return an error as a report.
            print(f"[FUNDAMENTALS] Analysis failed: {type(exc).__name__}")
            raise
    return fundamentals_analyst_node
