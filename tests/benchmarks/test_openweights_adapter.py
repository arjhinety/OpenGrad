from opengrad.benchmarks.adapters.openweights import (
    OpenWeightsAdapter,
    parse_openweights_reply,
    render_openweights_prompt,
)
from opengrad.benchmarks.backends.protocol import GenerationResult


def test_openweights_prompt_rendering_is_compact() -> None:
    tools = [
        {"name": "web_search", "description": "Search web.", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
        {"name": "read_file", "description": "Read file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
    ]
    prompt_bare = render_openweights_prompt(tools, "Find weather in Tokyo", format_mode="bare")
    assert '{"tool": "name", "arguments": {"argument": "value"}}' in prompt_bare
    # Must be under 150 words / lightweight without explicit instruction bloating
    assert len(prompt_bare.split()) < 150

    prompt_tagged = render_openweights_prompt(tools, "Find weather in Tokyo", format_mode="tagged")
    assert "<tool_call>" in prompt_tagged


def test_openweights_reply_parser() -> None:
    # 1. Bare JSON format
    dec1, name1, args1 = parse_openweights_reply('{"tool": "web_search", "arguments": {"query": "OpenWeights"}}')
    assert dec1 == "CALL"
    assert name1 == "web_search"
    assert args1 == {"query": "OpenWeights"}

    # 2. Tagged format
    dec2, name2, args2 = parse_openweights_reply('<tool_call>{"name": "fetch_url", "arguments": {"url": "https://example.com"}}</tool_call>')
    assert dec2 == "CALL"
    assert name2 == "fetch_url"
    assert args2 == {"url": "https://example.com"}

    # 3. Direct answer without tool
    dec3, name3, _args3 = parse_openweights_reply("Tokyo is the capital city of Japan.")
    assert dec3 == "ANSWER"
    assert name3 is None


def test_openweights_adapter_evaluation() -> None:
    adapter = OpenWeightsAdapter()
    tasks = adapter.load_tasks(split="bare", limit=4)
    assert len(tasks) == 4

    search_task = tasks[0]
    gen_call = GenerationResult(
        text='{"tool": "web_search", "arguments": {"query": "llama.cpp"}}',
        prompt_tokens=45,
        completion_tokens=15,
        latency=0.08,
    )
    res = adapter.evaluate_task(search_task, gen_call)
    assert res.success is True
    assert res.score == 1.0
    assert res.failure_category is None
