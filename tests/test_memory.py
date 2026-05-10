from unittest.mock import MagicMock, patch

from app.data_models import ChatMessage


def test_format_chat_history_empty():
    from app.chains import _format_chat_history

    assert _format_chat_history(None) == ""
    assert _format_chat_history([]) == ""


def test_format_chat_history_renders_turns():
    from app.chains import _format_chat_history

    history = [
        ChatMessage(human="What do customers say about pizza?", ai="They love it."),
        ChatMessage(human="And the crust?", ai="Mostly positive."),
    ]
    out = _format_chat_history(history)
    assert "Human: What do customers say about pizza?" in out
    assert "AI: They love it." in out
    assert "Human: And the crust?" in out
    assert "AI: Mostly positive." in out


@patch("app.chains.query_parser_llm")
def test_rewrite_question_skips_llm_on_empty_history(mock_llm):
    from app.chains import _rewrite_question

    out = _rewrite_question("What do customers say?", None)
    assert out == "What do customers say?"
    mock_llm.invoke.assert_not_called()

    out = _rewrite_question("What do customers say?", [])
    assert out == "What do customers say?"
    mock_llm.invoke.assert_not_called()


@patch("app.chains.query_parser_llm")
def test_rewrite_question_invokes_llm_with_history(mock_llm):
    from app.chains import _rewrite_question

    mock_response = MagicMock()
    mock_response.content = "What do customers say about the pizza crust?"
    mock_llm.invoke.return_value = mock_response

    history = [ChatMessage(human="What do customers say about pizza?", ai="They love it.")]
    out = _rewrite_question("What about the crust?", history)

    assert out == "What do customers say about the pizza crust?"
    mock_llm.invoke.assert_called_once()
    prompt_arg = mock_llm.invoke.call_args[0][0]
    assert "What about the crust?" in prompt_arg
    assert "What do customers say about pizza?" in prompt_arg


@patch("app.chains.query_parser_llm")
def test_rewrite_question_falls_back_on_llm_error(mock_llm):
    from app.chains import _rewrite_question

    mock_llm.invoke.side_effect = RuntimeError("LLM down")
    history = [ChatMessage(human="prior", ai="response")]
    out = _rewrite_question("follow-up", history)
    # Memory failures must not break the main flow.
    assert out == "follow-up"


@patch("app.chains.query_parser_llm")
def test_rewrite_question_falls_back_on_empty_response(mock_llm):
    from app.chains import _rewrite_question

    mock_response = MagicMock()
    mock_response.content = "   "
    mock_llm.invoke.return_value = mock_response

    history = [ChatMessage(human="prior", ai="response")]
    out = _rewrite_question("follow-up", history)
    assert out == "follow-up"
