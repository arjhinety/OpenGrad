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

## 4. Android Studio & Pixel Phone Provisioning

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
