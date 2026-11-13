from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

STRUCTURED_FORMAT_RULE = (
    "STRUCTURED OUTPUT CONTRACT: Use the supplied response schema for this call. "
    "If the provider supplies a schema function/tool, return its fields through that schema call; "
    "otherwise return only the JSON object required by the response schema. "
    "Do not substitute Markdown, code fences, bullet lists, or a standalone FINAL TRANSACTION PROPOSAL line. "
    "Output-format instructions in the decision context are for later human-readable rendering, "
    "not the format of this structured response. Put concise explanations in the schema's text fields. "
    "Preserve all decision and risk constraints; do not invent missing financial values."
)


def structured_prompt(prompt: Any) -> Any:
    """Disambiguate rendered string prompts only; leave other prompt types intact."""
    if not isinstance(prompt, str):
        return prompt
    return (STRUCTURED_FORMAT_RULE + "\n\nDECISION CONTEXT:\n" + prompt
            + "\n\n" + STRUCTURED_FORMAT_RULE)


def safe_error_label(exc: Exception) -> str:
    """Do not log provider bodies, rejected generations, prompts or credentials."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    allowed = ("json_validate_failed", "tool_use_failed", "invalid_request_error",
               "context_length_exceeded", "rate_limit_exceeded", "model_not_found")
    if isinstance(code, str) and code in allowed:
        return type(exc).__name__ + "/" + code
    return type(exc).__name__


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    try:
        return llm.with_structured_output(schema)
    except (AttributeError, NotImplementedError) as exc:
        logger.warning("%s structured output unavailable; using free text (%s)",
                       agent_name, safe_error_label(exc))
        return None


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    content, _ = invoke_structured_object_or_freetext(
        structured_llm, plain_llm, prompt, render, agent_name)
    return content


def invoke_structured_object_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> tuple[str, Optional[T]]:
    """Return rendered text and the parsed object; never fabricate one from prose."""
    if structured_llm is not None:
        try:
            structured_value = structured_llm.invoke(structured_prompt(prompt))
            return render(structured_value), structured_value
        except Exception as exc:
            logger.warning("%s structured output failed; retrying as free text (%s)",
                           agent_name, safe_error_label(exc))

    # Only the structured attempt gets the format override. The existing fallback
    # still uses the original human-readable prompt and returns no parsed object.
    response = plain_llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return content, None


def self_test() -> bool:
    import unittest
    from types import SimpleNamespace
    class Decision(BaseModel):
        action: str
    class Model:
        def __init__(self, value=None, error=None):
            self.value, self.error, self.calls = value, error, []
        def invoke(self, prompt):
            self.calls.append(prompt)
            if self.error is not None:
                raise self.error
            return self.value
    def render(value):
        return "FINAL TRANSACTION PROPOSAL: " + value.action
    class Tests(unittest.TestCase):
        def test_structured_call_gets_contract_and_keeps_decision_context(self):
            original = "Do not exceed risk limits. End with Markdown bullets."
            structured = Model(Decision(action="HOLD"))
            plain = Model()
            content, value = invoke_structured_object_or_freetext(structured, plain, original, render, "Unit Agent")
            self.assertIn(original, structured.calls[0])
            self.assertTrue(structured.calls[0].startswith(STRUCTURED_FORMAT_RULE))
            self.assertTrue(structured.calls[0].endswith(STRUCTURED_FORMAT_RULE))
            self.assertEqual(content, "FINAL TRANSACTION PROPOSAL: HOLD")
            self.assertIs(value, structured.value)
            self.assertEqual(plain.calls, [])
        def test_fallback_gets_original_prompt_and_no_structured_object(self):
            original = "Original free-text prompt"
            plain = Model(SimpleNamespace(content="FINAL TRANSACTION PROPOSAL: HOLD"))
            with self.assertLogs(logger, level="WARNING"):
                content, value = invoke_structured_object_or_freetext(
                    Model(error=RuntimeError("private-provider-value")), plain, original, render, "Unit Agent")
            self.assertEqual(plain.calls, [original])
            self.assertIsNone(value)
            self.assertIn("HOLD", content)
        def test_text_only_api_uses_same_structured_contract(self):
            structured = Model(Decision(action="HOLD"))
            content = invoke_structured_or_freetext(structured, Model(), "Context", render, "Unit Agent")
            self.assertEqual(content, "FINAL TRANSACTION PROPOSAL: HOLD")
            self.assertIn("STRUCTURED OUTPUT CONTRACT", structured.calls[0])
        def test_non_string_prompt_is_preserved(self):
            original = [SimpleNamespace(content="Existing message")]
            self.assertIs(structured_prompt(original), original)
        def test_provider_binding_has_no_new_api_parameters(self):
            class Binder:
                def with_structured_output(self, schema):
                    self.schema = schema
                    return "bound"
            binder = Binder()
            self.assertEqual(bind_structured(binder, Decision, "Unit Agent"), "bound")
            self.assertIs(binder.schema, Decision)
        def test_raw_provider_error_and_generation_not_logged(self):
            error = RuntimeError("private-secret-and-generated-plan")
            error.body = {"error": {"code": "json_validate_failed", "failed_generation": "private-account-value"}}
            with self.assertLogs(logger, level="WARNING") as logs:
                invoke_structured_object_or_freetext(Model(error=error),
                    Model(SimpleNamespace(content="HOLD")), "Context", render, "Unit Agent")
            text = " ".join(logs.output)
            self.assertIn("json_validate_failed", text)
            self.assertNotIn("private-", text)
        def test_unknown_provider_error_codes_not_logged(self):
            error = RuntimeError("private-message")
            error.body = {"code": "private-code"}
            self.assertEqual(safe_error_label(error), "RuntimeError")
        def test_plain_failure_propagates_without_another_retry(self):
            error = RuntimeError("plain request failure")
            plain = Model(error=error)
            with self.assertRaises(RuntimeError) as caught:
                invoke_structured_object_or_freetext(None, plain, "Context", render, "Unit Agent")
            self.assertIs(caught.exception, error)
            self.assertEqual(len(plain.calls), 1)
        def test_unavailable_binding_warning_is_sanitized(self):
            class Binder:
                def with_structured_output(self, schema):
                    raise NotImplementedError("private-endpoint")
            with self.assertLogs(logger, level="WARNING") as logs:
                self.assertIsNone(bind_structured(Binder(), Decision, "Unit Agent"))
            self.assertNotIn("private-endpoint", " ".join(logs.output))
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("Only --self-test is available as a command.")
    raise SystemExit(0 if self_test() else 1)
