<p align="center">

```shell
████████╗     ██████╗ ██████╗ ██████╗ ███████╗
╚══██╔══╝    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
   ██║       ██║     ██║   ██║██║  ██║█████╗
   ██║       ██║     ██║   ██║██║  ██║██╔══╝
   ██║       ╚██████╗╚██████╔╝██████╔╝███████╗
   ╚═╝        ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
```

</p>

<p align="center">
  A terminal-native AI coding agent. Written in Python. Provider-agnostic.
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#usage">Usage</a> |
  <a href="#providers">Providers</a> |
  <a href="#configuration">Configuration</a> |
  <a href="#architecture">Architecture</a>
</p>

---

tcode is an interactive AI coding assistant that runs entirely in your terminal. It connects to the LLM provider of your choice, executes tools on your behalf, and renders everything in a full-screen terminal UI with streaming responses, inline diffs, and permission controls.


<img src="https://github.com/Erickrus/tcode/blob/main/tcode.gif?raw=true" width=800px />

## tcode

- **Provider-agnostic.** Not locked to a single vendor. Use Anthropic, OpenAI, Google, Azure, Ollama, or any OpenAI-compatible endpoint. Switch models with a flag.
- **Terminal-native.** Full-screen TUI with streaming, inline diffs, shell mode, and native text selection. No browser, no Electron, no GUI dependencies.
- **Tool-enabled.** Built-in tools for file I/O, code search, shell execution, and HTTP requests. Extend with MCP servers for custom tooling.
- **Permission-aware.** Every sensitive action requires explicit approval. Always-allow rules for trusted operations. No surprises.
- **Session-persistent.** Conversations are stored in SQLite with structured message parts, token tracking, and cost accounting. Pick up where you left off.
- **Lightweight.** Pure Python. No Node.js runtime. No heavy frameworks. Starts in under a second.

## Installation

### Prerequisites

- Python 3.12+
- Git (for snapshot and diff features)

### From source

```bash
git clone https://github.com/Erickrus/tcode.git
cd tcode
python3.12 -m venv .
bin/pip install -r requirements.txt
```

### From source
```bash
pip install tcode
```

### Set up API keys

```bash
# Pick your provider
export TCODE_PROVIDER_ANTHROPIC_API_KEY="sk-ant-..."
export TCODE_PROVIDER_OPENAI_API_KEY="sk-..."
export TCODE_PROVIDER_GEMINI_API_KEY="..."
```

### Verify

```bash
bin/python3.12 tcode.py
```

You should see the tcode welcome screen with your configured model.

## Usage

### Interactive mode (default)

```bash
bin/python3.12 tcode.py
```

Launches the full-screen terminal UI. Type your prompt, press Enter, and watch the agent work.

### Single prompt

```bash
bin/python3.12 tcode.py "find and fix the memory leak in server.py"
```

Runs a single prompt, prints the result, and exits.

### REPL mode

```bash
bin/python3.12 tcode.py --repl
```

Classic read-eval-print loop without the full-screen UI.

### Model and provider override

```bash
bin/python3.12 tcode.py --provider anthropic --model claude-sonnet-4-20250514
bin/python3.12 tcode.py --provider openai --model gpt-4
bin/python3.12 tcode.py --provider ollama --model llama3
```

### Agent selection

```bash
bin/python3.12 tcode.py -a explore "how does the auth system work?"
bin/python3.12 tcode.py -a build "add rate limiting to the API"
```

## TUI Reference

### Commands

| Command    | Action                          |
|------------|---------------------------------|
| `/new`     | Start a new session             |
| `/compact` | Compress conversation history   |
| `/mcp`     | List MCP servers and tools      |
| `/memory`  | Show project memory             |
| `/memory compact` | Consolidate memory entries |
| `/help`    | Show help                       |
| `/quit`    | Exit                            |

### Keyboard

| Key              | Action                              |
|------------------|-------------------------------------|
| `Enter`          | Submit prompt                       |
| `Shift+Enter`    | Insert newline                      |
| `!`              | Toggle shell mode (on empty prompt) |
| `Escape` (x2)    | Cancel current agent run            |
| `Ctrl+C`         | Abort agent run                     |
| `Ctrl+Q`         | Force quit                          |
| `PageUp/PageDown`| Scroll message history              |
| `Up/Down`        | Navigate prompt history             |

### Shell mode

Type `!` on an empty prompt to switch to shell mode. The prefix changes from `>` to `$`. Commands are executed directly and output is displayed inline. Type `!` again to switch back.

### Inline diffs

When the agent edits a file, tcode renders a unified diff directly in the message stream:

```diff
  edit config.py [done]
  @@ -1,3 +1,3 @@
   import os
-  DEBUG = True
+  DEBUG = False
   PORT = 8080
```

Deletions appear on a red background. Additions on green. Context lines in neutral gray.

### Permissions

Sensitive operations (file writes, shell commands, etc.) trigger an inline permission prompt:

```
| Permission: builtin_shell
|   command: rm -rf build/
| [y] Allow  [n] Deny  [a] Always
```

Press `y` to allow once, `n` to deny, or `a` to always allow that operation type for the session.

## Providers

| Provider   | SDK                    | Streaming | Tool Calls |
|------------|------------------------|-----------|------------|
| Anthropic  | `anthropic`            | Yes       | Yes        |
| OpenAI     | `openai`               | Yes       | Yes        |
| Azure      | `openai` (Azure mode)  | Yes       | Yes        |
| Gemini     | `google-generativeai`  | Yes       | Yes        |
| Ollama     | `ollama`               | Yes       | Yes        |
| LiteLLM    | `openai` (compatible)  | Yes       | Yes        |

All adapters implement the same `ProviderAdapter` interface with async streaming via `ProviderChunk` protocol. Adding a new provider means implementing one class.

## Configuration

tcode reads configuration from three layers, merged in order:

1. **Global** `~/.config/tcode/tcode.json`
2. **Project** `./tcode.json` (or `.jsonc`)
3. **Environment variables**

### Example config

```json
{
  "model": {
    "providerId": "anthropic",
    "modelId": "claude-sonnet-4-20250514"
  },
  "providers": {
    "anthropic": {
      "apiKey": "{env:ANTHROPIC_API_KEY}"
    },
    "openai": {
      "apiKey": "{env:OPENAI_API_KEY}"
    },
    "ollama": {
      "baseUrl": "http://localhost:11434"
    }
  },
  "instructions": [
    "Always explain your reasoning before making changes.",
    "Prefer small, focused commits."
  ],
  "mcp": {
    "servers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
      }
    }
  }
}
```

### Environment variables

| Variable                       | Description                     |
|--------------------------------|---------------------------------|
| `TCODE_PROJECT_DIR`            | Project directory               |
| `TCODE_DB_PATH`                | SQLite database path            |
| `TCODE_MODEL_PROVIDER`         | Default provider                |
| `TCODE_MODEL_ID`               | Default model ID                |
| `TCODE_PROVIDER_*_API_KEY`     | API key for a provider          |
| `TCODE_PROVIDER_*_BASE_URL`    | Base URL for a provider         |

Variable substitution is supported in config files: `{env:VAR_NAME}` expands to the environment variable value, `{file:path}` reads from a file.

## MCP Integration

tcode supports the [Model Context Protocol](https://modelcontextprotocol.io/) for extending the agent with external tools and data sources.

Configure MCP servers in your `tcode.json`:

```json
{
  "mcp": {
    "servers": {
      "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {
          "GITHUB_TOKEN": "{env:GITHUB_TOKEN}"
        }
      }
    }
  }
}
```

Use `/mcp` in the TUI to see connected servers and available tools.

## Project Memory

tcode persists important context across sessions in `.tcode/MEMORY.md` — a human-readable, human-editable markdown file with timestamped entries.

Each session gets a lightweight index of memory titles in the system prompt. The agent calls `memory_read` or `memory_search` tools to fetch full details on demand. Memory tools run without permission prompts.

| Tool | Purpose |
|------|---------|
| `memory_write` | Save a new entry (or rewrite all with `replace_all`) |
| `memory_read` | Read full memory or entries matching a substring |
| `memory_search` | Search entries by keyword |
| `memory_delete` | Remove entries by title match |

Use `/memory` to view the file, `/memory compact` to consolidate entries via LLM.

## Architecture

```
tcode/
  agent.py              Agent loop: compose -> stream -> execute tools -> repeat
  cli.py                CLI entry point, argument parsing, REPL
  config.py             Layered configuration with env var substitution
  session.py            Structured message/part model, session lifecycle
  storage_sqlite.py     SQLite persistence with versioned schema
  tools.py              Tool registry and execution framework
  builtin_tools.py      File I/O, shell, HTTP, grep, edit, task management, memory
  memory.py             Project memory: parse, search, consolidate MEMORY.md
  mcp.py                MCP client wrapper with async streaming
  permissions.py        Async permission system with always-allow rules
  event.py              Event bus for real-time updates
  snapshot.py           Git-based file change tracking and restore
  server.py             FastAPI server for client/server mode
  providers/
    anthropic_adapter     Anthropic Claude (official SDK)
    openai_adapter        OpenAI GPT (official SDK)
    azure_openai_adapter  Azure OpenAI
    gemini_adapter        Google Gemini
    ollama_adapter        Ollama (local models)
    litellm_adapter       LiteLLM (OpenAI-compatible proxy)
  tui/
    screens/              Home and session screens
    widgets/              Prompt, message list, header, footer, diff rendering
    bridge.py             Connects TUI to core via event subscriptions
    state.py              Reactive state for UI synchronization
    theme.py              Color theme definitions

tcode_app.py            TUI application (runtui.App subclass)
```

The agent loop is the core: it composes messages from session history, streams a response from the LLM provider, executes any tool calls, and repeats until the model stops calling tools. Each step tracks token usage and cost. The event bus propagates state changes to the TUI for real-time rendering.

## Development

### Run tests

```bash
bin/python3.12 -m pytest tcode/tests/ -q
```

### Run with verbose tool logging

```bash
bin/python3.12 tcode.py --verbose
```

### Start the HTTP API server

```bash
bin/python3.12 -m uvicorn tcode.server:app --port 8080
```

## Dependencies

| Package              | Purpose                          |
|----------------------|----------------------------------|
| `pydantic`           | Type-safe models and validation  |
| `httpx`              | Async HTTP client                |
| `fastapi` + `uvicorn`| HTTP API server                  |
| `openai`             | OpenAI / Azure / LiteLLM SDK    |
| `anthropic`          | Anthropic Claude SDK             |
| `ollama`             | Ollama SDK                       |
| `google-generativeai`| Google Gemini SDK                |
| `runtui`             | Terminal UI framework            |

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>tcode</strong> -- AI-powered coding, from your terminal.
</p>
