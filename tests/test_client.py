from mi300x_bench.client import build_payload, chat_completions_url, models_url


def test_endpoint_urls() -> None:
    assert chat_completions_url("http://h:8000/v1") == "http://h:8000/v1/chat/completions"
    assert (
        chat_completions_url("http://h:8000/v1/chat/completions")
        == "http://h:8000/v1/chat/completions"
    )
    assert models_url("http://h:8000/v1") == "http://h:8000/v1/models"
    assert models_url("http://h:8000/v1/chat/completions") == "http://h:8000/v1/models"


def test_payload_omits_ignore_eos_unless_asked() -> None:
    bare = build_payload(model="m", prompt="p", output_len=3, temperature=0, ignore_eos=False)
    assert "ignore_eos" not in bare
    assert bare["max_tokens"] == 3
    assert bare["stream"] is True
    assert bare["stream_options"] == {"include_usage": True}
    flagged = build_payload(model="m", prompt="p", output_len=3, temperature=0, ignore_eos=True)
    assert flagged["ignore_eos"] is True
