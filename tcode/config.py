"""Configuration system for tcode.

Loads configuration from:
  1. Global: ~/.config/tcode/tcode.json (or tcode.jsonc)
  2. Project: ./tcode.json (or tcode.jsonc) in the working directory
  3. Environment variable overrides

Later sources override earlier sources (project > global > defaults).

Reference: opencode config/config.ts
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


# ---- Schema ----

class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class ModelRef(BaseModel):
    """Reference to a specific model on a specific provider."""
    provider_id: str = "litellm"
    model_id: str = "claude-haiku-4-5-20251001"


class AgentConfig(BaseModel):
    """Configuration for a named agent."""
    model: Optional[ModelRef] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    permission: Optional[List[Dict[str, Any]]] = None
    steps: Optional[int] = None
    mode: Optional[str] = None  # "primary" | "subagent"
    hidden: Optional[bool] = None
    description: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class McpServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    type: Optional[str] = None  # "http" | "local"; auto-detected if not set
    command: Optional[Union[str, List[str]]] = None  # for local: executable string or [exe, ...args]
    args: Optional[List[str]] = None  # for local: arguments when command is a string
    url: Optional[str] = None  # for http
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: int = 60
    enabled: bool = True

    @model_validator(mode='after')
    def _infer_type(self):
        if self.type is None:
            if self.command is not None:
                self.type = 'local'
            else:
                self.type = 'http'
        return self


class CommandConfig(BaseModel):
    """Configuration for a named command."""
    description: Optional[str] = None
    agent: Optional[str] = None
    model: Optional[str] = None
    template: str = ""
    subtask: bool = False


class SkillConfig(BaseModel):
    """Configuration for skill discovery."""
    paths: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class TcodeConfig(BaseModel):
    """Root configuration schema for tcode."""
    provider: Dict[str, ProviderConfig] = Field(default_factory=dict)
    model: ModelRef = Field(default_factory=ModelRef)
    agent: Dict[str, AgentConfig] = Field(default_factory=dict)
    mcp: Dict[str, McpServerConfig] = Field(default_factory=dict)
    command: Dict[str, CommandConfig] = Field(default_factory=dict)
    skill: SkillConfig = Field(default_factory=SkillConfig)
    permission: Dict[str, Any] = Field(default_factory=dict)
    instructions: List[str] = Field(default_factory=list)


# ---- JSONC stripping ----

def _strip_jsonc_comments(text: str) -> str:
    """Strip // and /* */ comments from JSONC text."""
    result = []
    i = 0
    in_string = False
    while i < len(text):
        ch = text[i]
        if in_string:
            result.append(ch)
            if ch == '\\' and i + 1 < len(text):
                result.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue
        if ch == '/' and i + 1 < len(text):
            next_ch = text[i + 1]
            if next_ch == '/':
                # line comment — skip to end of line
                while i < len(text) and text[i] != '\n':
                    i += 1
                continue
            if next_ch == '*':
                # block comment — skip to */
                i += 2
                while i + 1 < len(text) and not (text[i] == '*' and text[i + 1] == '/'):
                    i += 1
                i += 2  # skip */
                continue
        result.append(ch)
        i += 1
    return ''.join(result)


def _load_json_file(path: str) -> Dict[str, Any]:
    """Load a JSON or JSONC file, stripping comments if needed."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    if path.endswith('.jsonc'):
        text = _strip_jsonc_comments(text)
    return json.loads(text)


# ---- Environment variable substitution ----

_ENV_RE = re.compile(r'\{env:([^}]+)\}')
_FILE_RE = re.compile(r'\{file:([^}]+)\}')


def _substitute_vars(obj: Any, config_dir: str = '.') -> Any:
    """Recursively substitute {env:VAR} and {file:path} in string values."""
    if isinstance(obj, str):
        # Replace {env:VAR_NAME}
        def _env_repl(m):
            return os.environ.get(m.group(1), '')
        result = _ENV_RE.sub(_env_repl, obj)
        # Replace {file:path}
        def _file_repl(m):
            fpath = m.group(1)
            if fpath.startswith('~/'):
                fpath = os.path.expanduser(fpath)
            elif not os.path.isabs(fpath):
                fpath = os.path.join(config_dir, fpath)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                return ''
        result = _FILE_RE.sub(_file_repl, result)
        return result
    elif isinstance(obj, dict):
        return {k: _substitute_vars(v, config_dir) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_vars(v, config_dir) for v in obj]
    return obj


# ---- Merge ----

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override into base. Lists are concatenated for 'instructions',
    replaced for everything else. Dicts are merged recursively."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key == 'instructions' and isinstance(result.get(key), list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value
    return result


# ---- Loading ----

def _find_config_file(directory: str) -> Optional[str]:
    """Find tcode.json or tcode.jsonc in a directory."""
    for name in ('tcode.json', 'tcode.jsonc'):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def _load_from_directory(directory: str) -> Dict[str, Any]:
    """Load config from a directory, returns raw dict."""
    path = _find_config_file(directory)
    if not path:
        return {}
    try:
        raw = _load_json_file(path)
        return _substitute_vars(raw, directory)
    except Exception:
        return {}


def _env_overrides() -> Dict[str, Any]:
    """Extract config overrides from environment variables.

    Supported:
      TCODE_PROVIDER_<NAME>_API_KEY -> provider.<name>.api_key
      TCODE_PROVIDER_<NAME>_BASE_URL -> provider.<name>.base_url
      TCODE_MODEL_PROVIDER -> model.provider_id
      TCODE_MODEL_ID -> model.model_id
    """
    overrides: Dict[str, Any] = {}

    # Provider overrides — match known suffixes
    prefix = 'TCODE_PROVIDER_'
    _suffixes = {
        '_API_KEY': 'api_key',
        '_BASE_URL': 'base_url',
        '_APIKEY': 'api_key',
        '_BASEURL': 'base_url',
    }
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        for suffix, field_name in _suffixes.items():
            if rest.endswith(suffix):
                provider_name = rest[:-len(suffix)].lower()
                if provider_name:
                    overrides.setdefault('provider', {}).setdefault(provider_name, {})[field_name] = val
                break

    # Model overrides
    mp = os.environ.get('TCODE_MODEL_PROVIDER')
    mi = os.environ.get('TCODE_MODEL_ID')
    if mp or mi:
        overrides['model'] = {}
        if mp:
            overrides['model']['provider_id'] = mp
        if mi:
            overrides['model']['model_id'] = mi

    return overrides


def load_config(project_dir: Optional[str] = None) -> TcodeConfig:
    """Load configuration with precedence: global -> project -> env overrides.

    Args:
        project_dir: Project directory to search for tcode.json. Defaults to cwd.
    """
    if project_dir is None:
        project_dir = os.getcwd()

    # 1. Global config
    global_dir = os.path.join(os.path.expanduser('~'), '.config', 'tcode')
    global_raw = _load_from_directory(global_dir)

    # 2. Project config
    project_raw = _load_from_directory(project_dir)

    # 3. Merge: global -> project
    merged = _deep_merge(global_raw, project_raw)

    # 4. Env overrides
    env = _env_overrides()
    if env:
        merged = _deep_merge(merged, env)

    # 5. Validate with Pydantic
    return TcodeConfig.model_validate(merged)


# ---- Global config singleton ----

_config: Optional[TcodeConfig] = None
_project_dir: Optional[str] = None


def get_config() -> TcodeConfig:
    """Get the current configuration (lazy-loaded singleton)."""
    global _config
    if _config is None:
        _config = load_config(_project_dir)
    return _config


def set_project_dir(directory: str):
    """Set the project directory and invalidate cached config."""
    global _project_dir, _config
    _project_dir = directory
    _config = None


def reload_config():
    """Force reload of configuration."""
    global _config
    _config = None
    return get_config()


def get_provider_config(provider_id: str) -> ProviderConfig:
    """Get provider config, falling back to empty defaults."""
    cfg = get_config()
    return cfg.provider.get(provider_id, ProviderConfig())


def get_default_model() -> ModelRef:
    """Get the default model reference."""
    return get_config().model


def save_model_to_project(project_dir: str, provider_id: str, model_id: str) -> None:
    """Save model selection to the project's tcode.json.

    Reads the existing file, updates only the 'model' key, and writes back.
    Creates the file if it doesn't exist.
    """
    path = os.path.join(project_dir, 'tcode.json')
    raw: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            raw = _load_json_file(path)
        except Exception:
            raw = {}
    raw['model'] = {'provider_id': provider_id, 'model_id': model_id}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=2)
        f.write('\n')
    # Invalidate cached config so next get_config() picks up the change
    global _config
    _config = None
