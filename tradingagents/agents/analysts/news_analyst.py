from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
import json
from tradingagents.prompts import load_prompt, render_prompt

try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def _news_tool_calls(message):
    """Validate calls without rebuilding or mutating the provider's AI message."""
    if getattr(message, "invalid_tool_calls", None):
        raise ValueError("NEWS_INVALID_TOOL_CALL")
    calls = getattr(message, "tool_calls", None)
    if not calls:
        calls = (getattr(message, "additional_kwargs", {}) or {}).get("tool_calls", [])
    if calls is None:
        calls = []
    if not isinstance(calls, (list, tuple)):
        raise ValueError("NEWS_INVALID_TOOL_CALL_LIST")
    parsed, ids = [], set()
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("NEWS_INVALID_TOOL_CALL")
        function = call.get("function") or {}
        if not isinstance(function, dict):
            raise ValueError("NEWS_INVALID_TOOL_CALL")
        name = call.get("name") or function.get("name")
        call_id = call.get("id")
        args = call["args"] if "args" in call else function.get("arguments", {})
        if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
            raise ValueError("NEWS_INVALID_TOOL_NAME_OR_ID")
        if call_id in ids:
            raise ValueError("NEWS_DUPLICATE_TOOL_CALL_ID")
        ids.add(call_id)
        try:
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError()
            args = json.loads(json.dumps(args, allow_nan=False))
        except (TypeError, ValueError):
            raise ValueError("NEWS_INVALID_TOOL_ARGUMENTS") from None
        parsed.append({"name": name, "id": call_id, "args": args})
    return parsed


def _news_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content if isinstance(block, dict)
                         and block.get("type") in ("text", "output_text")
                         and isinstance(block.get("text"), str)).strip()
    return ""


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        is_crypto = "/" in ticker or "USD" in ticker.upper() or "USDT" in ticker.upper()
        openai_available = toolkit.has_openai_web_search()
        finnhub_available = toolkit.has_finnhub()
        coindesk_available = toolkit.has_coindesk()
        global_news_available = (toolkit.config["online_tools"] and openai_available
                                 and bool(toolkit.config.get("news_global_openai_enabled", False)))
        tools = [toolkit.get_google_news]
        if global_news_available:
            tools.append(toolkit.get_global_news_openai)
        if is_crypto:
            if coindesk_available:
                tools.append(toolkit.get_coindesk_news)
        elif finnhub_available:
            tools.append(toolkit.get_finnhub_news_recent)
        source_labels = ["Google News"]
        if global_news_available:
            source_labels.append("OpenAI global web search")
        if is_crypto and coindesk_available:
            source_labels.append("CoinDesk/CryptoCompare")
        if not is_crypto and finnhub_available:
            source_labels.append("Finnhub")
        global_news_guidance = (
            f"**IMPORTANT:** When using get_global_news_openai, ALWAYS pass ticker_context='{ticker}' to get "
            + ("crypto-relevant global news (regulation, institutional adoption, DeFi developments)"
               if is_crypto else "sector-relevant global news")
            + " instead of generic macro news.\n" if global_news_available else "")
        source_guidance = (
            " Use all currently available news tools before concluding."
            f" Active sources: {', '.join(source_labels)}."
            " For `get_finnhub_news_recent`, pass ticker and curr_date from context."
            " Do not request broad macro/global web searches unless OpenAI global web search is listed as an active source;"
            " the Macro analyst handles that context when enabled."
        )
        system_message = render_prompt("analysts/news_system", ticker=ticker,
            global_news_guidance=global_news_guidance, source_guidance=source_guidance)
        asset_context = f"We are looking at the ticker: {ticker}"
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
            capture_agent_prompt("news_report", complete_prompt, ticker)
        except Exception as exc:
            print(f"[NEWS] Prompt capture unavailable: {type(exc).__name__}")
            capture_agent_prompt("news_report", system_message, ticker)

        chain = prompt | (llm.bind_tools(tools) if tools else llm)
        messages_history = list(state["messages"])
        result = chain.invoke(messages_history)
        max_tool_iterations = int(toolkit.config.get("max_tool_iterations_per_agent", 8))
        max_same_call_repeats = int(toolkit.config.get("max_same_tool_call_repeats", 1))
        tool_call_counts, tool_result_cache = {}, {}
        iteration_count = 0
        calls = _news_tool_calls(result)
        tool_map = {tool.name: tool for tool in tools}
        used_ids = set()
        while tools and calls and iteration_count < max_tool_iterations:
            for call in calls:
                if call["name"] not in tool_map:
                    raise ValueError("NEWS_UNKNOWN_TOOL")
                if call["id"] in used_ids:
                    raise ValueError("NEWS_REUSED_TOOL_CALL_ID")
            used_ids.update(call["id"] for call in calls)
            iteration_count += 1
            # One original AI message, followed by every corresponding tool result.
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
                        tool_result = f"Error running tool '{tool_name}': {type(tool_err).__name__}. No usable result from this call."
                messages_history.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
            result = chain.invoke(messages_history)
            calls = _news_tool_calls(result)

        if tools and calls:
            result = AIMessage(content=(_news_text(result.content)
                + f"\n\nTool-loop halted after {max_tool_iterations} iterations to prevent endless retries.").strip())
        analysis_content = _news_text(result.content)
        if not analysis_content:
            analysis_content = (
                "The analyst did not return a complete news narrative. "
                "State that limitation clearly in the final recommendation."
            )
        if "FINAL TRANSACTION PROPOSAL:" not in analysis_content:
            final_prompt = render_prompt("analysts/news_final_recommendation", ticker=ticker,
                                         analysis_content=analysis_content)
            final_result = llm.invoke(final_prompt)
            final_content = _news_text(final_result.content) if hasattr(final_result, "content") else str(final_result)
            analysis_content += "\n\n" + final_content
        result = AIMessage(content=analysis_content)
        return {"messages": [result], "news_report": result.content}
    return news_analyst_node
