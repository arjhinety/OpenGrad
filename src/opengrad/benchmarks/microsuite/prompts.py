"""Internal deterministic performance microsuite with frozen prompts and hashes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MicrosuitePrompt:
    id: str
    name: str
    category: str
    prompt: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    expected_type: str = "direct"
    sha256: str = ""

    def __post_init__(self) -> None:
        computed = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        if not self.sha256:
            object.__setattr__(self, "sha256", computed)
        elif self.sha256 != computed:
            raise ValueError(
                f"Prompt hash mismatch for {self.id}: expected {self.sha256}, got {computed}"
            )


# 10 Frozen Prompts representing controlled inference profiles (Section 16)
PROMPTS: list[MicrosuitePrompt] = [
    MicrosuitePrompt(
        id="micro_01_natural_language",
        name="Plain Natural Language",
        category="natural_language",
        prompt="Explain the key differences between synchronous and asynchronous I/O in three concise paragraphs.",
        sha256="0f99cb6106d7bdc91e50000b857446f739c5d769ca27ab4071a47618689e0933",
        expected_type="direct",
    ),
    MicrosuitePrompt(
        id="micro_02_long_form",
        name="Long-Form Generation",
        category="long_form",
        prompt="Write a detailed technical retrospective on how distributed consensus protocols evolved from Paxos to Raft and modern Byzantine Fault Tolerant variants.",
        sha256="203aafa8677cce1f6a9fcb38785726d9a60fc6dffb7a492a30a99415a0180ec7",
        expected_type="direct",
    ),
    MicrosuitePrompt(
        id="micro_03_python_coding",
        name="Python Coding",
        category="python_coding",
        prompt="Write an efficient, thread-safe LRU cache in Python using a doubly linked list and a hash map. Include type hints and docstrings.",
        sha256="1dd087071f2e59815f6e1be4900be229b9050fa362f9284e88a855af6cb51c51",
        expected_type="coding",
    ),
    MicrosuitePrompt(
        id="micro_04_json_generation",
        name="JSON Generation",
        category="json_generation",
        prompt="Produce a strictly valid JSON schema and a matching JSON object representing a cloud compute instance specification with CPU, RAM, disk, and network interfaces.",
        sha256="b3e230fd29117ec6d9ebc4a03b4f167e69b6b59988109a459536d64bae94f01c",
        expected_type="json",
    ),
    MicrosuitePrompt(
        id="micro_05_single_tool_call",
        name="Single Tool Call",
        category="single_tool_call",
        prompt="Find the latest status and resource usage of container 'prod-api-worker-01'.",
        tools=[
            {
                "name": "get_container_status",
                "description": "Inspect container status and resource telemetry.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_id": {"type": "string", "description": "The target container ID or name"},
                        "include_metrics": {"type": "boolean", "description": "Whether to include memory/CPU usage"}
                    },
                    "required": ["container_id"]
                }
            }
        ],
        sha256="676ba96eb1bd73901fb67af13404c963ac41e19151d17e91a43fbe23a9acf517",
        expected_type="tool_call",
    ),
    MicrosuitePrompt(
        id="micro_06_parallel_tool_calls",
        name="Parallel Tool Calls",
        category="parallel_tool_calls",
        prompt="Check the current weather in both Tokyo and Seattle simultaneously.",
        tools=[
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        ],
        sha256="c2a612c173770ebbfa74fe8a259c87bed4725cb8fb77f2b6386008f92fa0555a",
        expected_type="tool_call",
    ),
    MicrosuitePrompt(
        id="micro_07_multi_turn_tool",
        name="Multi-Turn Tool Interaction",
        category="multi_turn_tool",
        prompt="User: Query order #84920.\nAssistant: <tool_call>{\"name\":\"lookup_order\",\"arguments\":{\"order_id\":\"84920\"}}</tool_call>\nObservation: {\"status\":\"shipped\",\"tracking\":\"TRK9921\"}\nUser: When is it expected to arrive?",
        tools=[
            {
                "name": "lookup_tracking_estimate",
                "description": "Get estimated delivery date from tracking number.",
                "parameters": {
                    "type": "object",
                    "properties": {"tracking_number": {"type": "string"}},
                    "required": ["tracking_number"]
                }
            }
        ],
        sha256="b19eb367b406153c9ab84cce19d218b057d181d66d818454288179fabbf3b75e",
        expected_type="tool_call",
    ),
    MicrosuitePrompt(
        id="micro_08_long_context_selection",
        name="Long-Context Tool Selection",
        category="long_context_selection",
        prompt="Given a large catalog of 10 microservice endpoints (auth, billing, inventory, notifications, analytics, metrics, storage, compute, dns, telemetry), identify and invoke the single correct endpoint to issue a refund of $45.00 for payment 'pay_7721'.",
        tools=[
            {"name": f"service_endpoint_{i}", "parameters": {"type": "object", "properties": {}}}
            for i in range(10)
        ] + [
            {
                "name": "process_refund",
                "description": "Refund an existing payment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payment_id": {"type": "string"},
                        "amount": {"type": "number"}
                    },
                    "required": ["payment_id", "amount"]
                }
            }
        ],
        sha256="ef1e6d565864e9e523c628a1f84411b88e6b9ad588f4f5eea1e75c1cdc9a0700",
        expected_type="tool_call",
    ),
    MicrosuitePrompt(
        id="micro_09_repetitive_structured",
        name="Repetitive Structured Generation",
        category="repetitive_structured",
        prompt="Generate a markdown table of HTTP status codes from 200 to 206 with columns: Code, Name, Description, Idempotent (Yes/No).",
        sha256="7404281977bdd295677dba15ed353c9e6fe20ed7947098bc86fae8cd9601d41c",
        expected_type="structured",
    ),
    MicrosuitePrompt(
        id="micro_10_reasoning_heavy",
        name="Reasoning-Heavy Generation",
        category="reasoning_heavy",
        prompt="Analyze the following race condition scenario: Process A reads variable X, Process B increments X, Process A writes back X+1. Walk step-by-step through why this results in a lost update and propose two distinct synchronization mechanisms to prevent it.",
        sha256="58385632460e4a0f19b1ecbfffc1c9f1361f607b4b09a61c0671dc507b8519c4",
        expected_type="reasoning",
    ),
]


def get_microsuite_prompts() -> list[MicrosuitePrompt]:
    return list(PROMPTS)


def parse_microsuite_output(text: str) -> dict[str, Any]:
    return {"text": text, "valid": bool(text.strip())}
