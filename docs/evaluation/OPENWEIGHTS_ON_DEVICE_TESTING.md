# On-Device Tool Calling & Android Studio Testing with OpenWeights

**Building in Public.** OpenGrad connects directly to practical on-device deployment questions exposed by [OpenWeights](https://github.com/alpharomercoma/openweights), an independent mobile open-weight model deployment engine developed by Alpha Romer Coma (Experimental Machines).

---

## 1. Why On-Device Testing Matters

On a mobile phone (e.g., Google Pixel 7, Galaxy S24, or iPhone), conventional large language model serving assumptions break down:
- **System Instruction Bloat**: Standard SFT models rely on 1,500–2,000 token system instructions describing functions. On an on-device NPU/CPU, prefilling 2,000 tokens takes 15–20 seconds, severely degrading interactive time to first token (TTFT) and consuming constrained RAM.
- **Strict Format Adherence**: Models must invoke tools reliably with concise prompt envelopes (<150 tokens) and clean syntax.

OpenGrad evaluates whether post-trained small models (Qwen3.5-2B) can execute tool calling effectively **without heavy prompt engineering**.

---

## 2. OpenWeights Tool Calling Architecture

OpenWeights (`core/tools/src/main/kotlin/io/github/alpharomercoma/openweights/core/tools/ToolPrompting.kt`) specifies two on-device tool calling arms:

1. **`CallFormat.BARE`**: Flat JSON object:
   ```json
   {"tool": "web_search", "arguments": {"query": "OpenGrad research"}}
   ```
2. **`CallFormat.TAGGED`**: Enclosed in tags:
   ```xml
   <tool_call>{"name": "fetch_url", "arguments": {"url": "https://example.com"}}</tool_call>
   ```

### Minimal Prompt Contract (<150 tokens):
```text
You can use these tools:
- web_search: Search the web for fresh information. Arguments: {"query":"string"}
- read_file: Read file contents from workspace. Arguments: {"path":"string"}

To use one, reply with only this and nothing else:
{"tool": "name", "arguments": {"argument": "value"}}
Do not explain that you are going to use it. Just send the object. If no tool is needed, answer normally and send no object.
```

---

## 3. Tool Calling Benchmark in OpenGrad

OpenGrad provides the `openweights` benchmark adapter (`src/opengrad/benchmarks/adapters/openweights.py`):
- Config: `configs/benchmarks/openweights.yaml`
- Evaluates the 18 standard OpenWeights tools: `web_search`, `fetch_url`, `read_file`, `write_file`, `list_dir`, `run_script`, `remember`, `recall`, `forget`, `create_plan`, `update_step`, `set_goal`, `report_outcome`, `ask_user`, `canvas_render`, etc.
- Measures: tool selection accuracy, direct answers when no tool is needed, format compliance, and prose leakage penalties.

Run on-device benchmark dry-run:
```bash
opengrad benchmark run --benchmark openweights --dry-run
```

---

## 4. Tool-call format contract with OpenWeights

OpenGrad and OpenWeights must agree on what a tool call *is*, or a checkpoint measured in one place is not comparable to the same checkpoint measured in the other. The two parsers are kept byte-compatible on purpose.

The format question has three parts, and conflating them is the failure mode:

| Layer | Who decides | Qwen3.5-2B |
| --- | --- | --- |
| What the model is *asked* for | The OpenWeights arm's prompt (`CallFormat.BARE` / `TAGGED`) | A request, not a constraint |
| What the model's template *allows* | The model's own chat template | XML `<function=...><parameter=...>` only |
| What gets *read back* | llama.cpp's parser, then `ToolCallParser`, then the adapter | XML branch of `ToolCallParser` |

When a model's own template carries tools, OpenWeights prefers it — `LlamaCppEngine` asks `nativeSupportsTools` of the loaded GGUF, and `PromptTemplates.forModel` refuses families whose protocol it has not transcribed (Qwen3.5 sits in that refusal list alongside vision and coder variants). So an arm's prompt cannot override a template that already fixes the shape, and the adapter must accept the native form rather than score it a format error.

OpenGrad's `parse_qwen_native_output` reads both the native XML form and the older JSON spelling, using the same `FUNCTION_TAG` / `PARAMETER_TAG` regexes as `ToolCallParser.parseTaggedXml`. `parse_openweights_reply` accepts the native XML shape first, then the two JSON arms.

Parameter values are kept as the strings the model emitted. The template stringifies every argument when rendering, so the original JSON type is not recoverable from a rendered transcript and is not guessed; argument comparison is an evaluator concern. `tests/benchmarks/test_openweights_format_contract.py` pins the shared grammar using OpenWeights' own fixtures.

Two upstream findings came out of this cross-check and are recorded rather than silently worked around:

- `alpharomercoma/openweights#2` — `parseTaggedXml` read only the first envelope and the first `<function>`, so a model calling twice in one reply lost the second call. Fixed and tested in that PR.
- A reply mixing an XML envelope with a JSON envelope still yields only the XML calls, because `ToolCallParser.parse` is a fallback chain and the first branch to claim the reply wins. Left as an upstream design question; OpenGrad does not depend on the mixed shape.

---

## 5. Android Studio & Pixel Phone Provisioning

The local host environment is provisioned with official Android development and testing tools:

- **Android Studio**: Installed at `/opt/android-studio/` (`/opt/android-studio/bin/studio.sh` with bundled JBR 21).
- **Android SDK & Platform Tools**: Configured at `/opt/android-sdk/` with `adb` at `/opt/android-sdk/platform-tools/adb`.
- **Pixel 7 Virtual Device**: Provisioned at `/root/.android/avd/pixel_phone.avd`:
  - Profile: Google Pixel 7 (`pixel_7`)
  - Target: Android 14.0 ("UpsideDownCake") API 34
  - Architecture: `x86_64`

Verify the provisioned AVD:
```bash
/opt/android-sdk/cmdline-tools/latest/bin/avdmanager list avd
```

Verify system doctor diagnostics:
```bash
opengrad doctor
```
