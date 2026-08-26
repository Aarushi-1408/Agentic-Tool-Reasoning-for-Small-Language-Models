# Minimal, Model-Agnostic Tool-Use Harness (`tool_agent`)

A clean, modular, and lightweight tool-use execution environment built for research and agent-training projects. Designed initially for `Qwen/Qwen3-1.7B` and seamless scaling to `Qwen/Qwen3-4B` or other model backends.

---

## Key Design Principles

1. **Thin Execution Layer**: The harness performs ZERO reasoning, keyword matching, heuristic decision-making, or planning outside the model.
2. **Model Autonomy**: The model independently decides:
   - Whether to call a tool or produce a direct answer.
   - Which tool to select from available schemas.
   - What arguments to pass to the tool.
   - How to interpret returned observations or error messages.
   - When to make follow-up tool calls vs when the task is complete.
3. **Structured Observation Feedback**: Invalid syntax, unknown tool calls, or bad argument types return structured `TOOL_ERROR` or `PARSE_ERROR` observations back to the model rather than silently guessing or fixing parameters.
4. **Trajectory Logging**: Captures full conversation trajectories (inputs, tool calls, arguments, observations, final answers, and termination reasons) suitable for SFT, RL (GRPO/PPO), or failure analysis.

---

## Repository Structure

```text
tool_agent/
│
├── README.md               # Project documentation and architectural explanation
├── requirements.txt        # Python dependency requirements
├── .gitignore              # Git ignore configuration
│
├── src/                    # Core harness implementation
│   ├── __init__.py
│   ├── model.py            # ModelInterface abstraction (HF Transformers & Mock backends)
│   ├── tools.py            # Generic Tool base class & ToolRegistry
│   ├── tool_schema.py      # Standard JSON Schema generator & validation utilities
│   ├── parser.py           # ModelOutputParser (ToolCallAction, FinalAnswerAction, ParseErrorAction)
│   ├── harness.py          # AgentHarness loop enforcing max_steps & thin execution semantics
│   ├── trajectory.py       # Trajectory and TrajectoryStep tracking & serialization
│   └── evaluator.py        # Evaluator module for measuring exact match, step counts, & tool validity
│
├── tools/                  # Registered tools directory
│   ├── __init__.py
│   └── example_tools.py    # DEMO ONLY tools (CalculatorTool & EchoTool)
│
├── configs/
│   └── default.yaml        # Configuration for model name, device, dtype, and agent max_steps
│
├── tasks/
│   └── example_tasks.json  # Benchmark dataset containing tool-requiring and direct-answer tasks
│
├── scripts/
│   ├── run_agent.py        # CLI to execute single task prompts with interactive trajectory logs
│   └── evaluate.py         # CLI to run batch evaluation on task datasets
│
└── tests/                  # Pytest unit test suite
    ├── test_parser.py      # Tests for tool call and final answer parsing
    ├── test_tools.py       # Tests for registry, unknown tools, and parameter validation
    └── test_harness.py     # End-to-end tests for harness loop, error recovery, and step limits
```

---

## Installation Commands

```bash
# Clone or navigate to the project directory
cd tool_agent

# Create and activate a Python virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Execution Commands

### 1. Run standard Qwen3-1.7B Model

```bash
python scripts/run_agent.py --task "Calculate 25 * 37." --config configs/default.yaml
```

To switch to `Qwen/Qwen3-4B`, update `configs/default.yaml`:
```yaml
model:
  name: Qwen/Qwen3-4B
```

### 2. Fast Offline Testing with Mock Backend

```bash
python scripts/run_agent.py --task "Calculate 25 * 37." --mock
python scripts/run_agent.py --task "What is the capital of France?" --mock
```

### 3. Run Frozen Benchmark Evaluation

```bash
# Fast offline evaluation using Mock backend on 60-task benchmark
python scripts/evaluate.py --tasks tasks/eval_v1.json --mock --output-dir evaluation/mock_baseline/

# Baseline evaluation on HuggingFace Qwen3-1.7B
python scripts/evaluate.py --tasks tasks/eval_v1.json --config configs/default.yaml --output-dir evaluation/baseline/
```

### 4. Run Pytest Unit Test Suite

```bash
python -m pytest tests
```

---

## Frozen Evaluation Benchmark (`tasks/eval_v1.json`)

The benchmark is a static, frozen 60-task dataset used for before/after fine-tuning evaluation.

### Task Category Breakdown

| Category | Tasks | % Share | Description |
| :--- | :---: | :---: | :--- |
| `no_tool` | 12 | 20% | Direct questions testing whether the model knows when NOT to use a tool. |
| `single_tool` | 15 | 25% | Single tool call tasks (`calculator`, `echo`, `lookup`, `string_manipulator`). |
| `two_step` | 15 | 25% | Multi-step tasks requiring 2 sequential tool operations. |
| `multi_step` | 9 | 15% | Longer multi-step workflows (3–4 tool calls). |
| `tool_selection` | 6 | 10% | Choosing between multiple available tools and argument extractions. |
| `error_recovery` | 3 | 5% | Recovering from `TOOL_ERROR` or syntax issues autonomously. |

### Calculated Evaluation Metrics

- **Overall Task Success Rate**: Percentage of total tasks answered correctly.
- **Tool Tasks Accuracy**: Accuracy on tasks requiring tool calls.
- **No-Tool Tasks Accuracy**: Accuracy on direct-answer tasks.
- **Tool Selection Accuracy**: Rate at which appropriate tools were selected.
- **Unnecessary Tool-Use Rate**: Percentage of `no_tool` tasks where tools were unnecessarily invoked.
- **Multi-Step Tasks Accuracy**: Success rate on two-step and multi-step tasks.
- **Valid Tool-Call Rate**: Ratio of error-free tool invocations.
- **Average Steps / Task**: Mean interaction steps per task.
- **Trajectory Logging**: Full trajectories saved line-by-line to `evaluation/<run_name>/trajectories.jsonl`.

---

## Architectural Breakdown

### 1. How the Model is Loaded
In `src/model.py`, the `HuggingFaceModel` wraps `AutoModelForCausalLM` and `AutoTokenizer` from Hugging Face `transformers`. The model name (`Qwen/Qwen3-1.7B`), torch device, dtype, and `max_new_tokens` are injected directly from `configs/default.yaml`. Changing the model name in YAML is all that is needed to test `Qwen/Qwen3-4B`.

### 2. How Tools are Registered
Tools inherit from `Tool` in `src/tools.py` and define their `name`, `description`, `parameters` (JSON Schema format), and `execute(arguments)` function. The `ToolRegistry` holds active tools and exposes `.schemas()`. The registry purely stores tools and performs parameter type validation—it **never** selects tools.

### 3. How Tool Calls are Parsed
The `ModelOutputParser` in `src/parser.py` inspects raw model output strings for:
- JSON tool call format: `{"type": "tool_call", "name": "...", "arguments": {...}}`
- Native Qwen `<tool_call>...</tool_call>` tag format
- JSON final answer format: `{"type": "final", "content": "..."}`
- Plain-text response (treated as a final answer)

If syntax or required fields are missing, it returns a `ParseErrorAction`. The parser does **not** alter parameters or fix model mistakes.

### 4. How Tools are Executed
When a `ToolCallAction` is parsed, the harness calls `tool_registry.execute(tool_name, arguments)`:
- If `tool_name` is unknown, it returns `TOOL_ERROR: Unknown tool: '...'`.
- If `arguments` fail JSON schema validation, it returns `TOOL_ERROR: Invalid arguments ...`.
- Otherwise, `tool.execute(arguments)` is invoked safely and returns `TOOL_RESULT: <output>`.

### 5. How Observations are Returned to the Model
Tool execution results or structured error strings are appended to the conversation message history as user role observations (e.g. `{"role": "user", "content": "TOOL_RESULT: 925"}`). The updated message list is then passed back to `model.generate()` in the next step.

### 6. How the Loop Terminates
The harness loop in `src/harness.py` continues until:
- The model outputs a `FinalAnswerAction` (`termination_reason = "final_answer"`).
- The maximum step limit is reached (`termination_reason = "max_steps_exceeded"`).

### 7. Where Actual Model Decision-Making Occurs
All decision-making occurs **inside the model weights**. The harness acts purely as an executor and message router.

---

## Running the SFT Model

To run inference or evaluate the fine-tuned SFT checkpoint (`Bitsbrains/aarushi-qwen3-1.7b-tool-agent-sft-v2`):

1. **Download the model weights from Hugging Face**:
   Download the trained SFT checkpoint files from Hugging Face Hub.

2. **Place the checkpoint files under local directory**:
   Save the model files into the directory structure:
   ```text
   models/qwen3-1.7b-tool-agent-sft-v2/
   ├── model.safetensors
   ├── config.json
   ├── generation_config.json
   ├── tokenizer.json
   ├── tokenizer_config.json
   └── chat_template.jinja
   ```

3. **Run 3-Task Smoke Test**:
   ```bash
   python scripts/smoke_test.py --model-path models/qwen3-1.7b-tool-agent-sft-v2/
   # Or directly from HuggingFace Hub ID:
   python scripts/smoke_test.py --model-path Bitsbrains/aarushi-qwen3-1.7b-tool-agent-sft-v2/
   ```

4. **Run Single Task Inference**:
   ```bash
   python scripts/run_agent.py --task "Calculate 25 * 37." --model-path models/qwen3-1.7b-tool-agent-sft-v2/
   ```

5. **Run Benchmark Evaluation (60 Frozen Tasks)**:
   ```bash
   python scripts/evaluate.py --tasks tasks/eval_v1.json --model-path models/qwen3-1.7b-tool-agent-sft-v2/ --output-dir evaluation/sft_v2_100/
   ```

