"""Tests for tcode configuration system."""
from __future__ import annotations
import json
import os
import tempfile
import pytest
from tcode.config import (
    TcodeConfig, ProviderConfig, ModelRef, AgentConfig,
    McpServerConfig, CommandConfig, SkillConfig,
    load_config, _strip_jsonc_comments, _deep_merge,
    _env_overrides, _load_json_file, _substitute_vars,
    get_config, set_project_dir, reload_config,
    get_provider_config, get_default_model,
)


# ---- JSONC stripping ----

def test_strip_jsonc_line_comments():
    text = '{"key": "value"} // this is a comment\n{"b": 1}'
    result = _strip_jsonc_comments(text)
    assert '//' not in result
    assert '"key": "value"' in result


def test_strip_jsonc_block_comments():
    text = '{"key": /* block comment */ "value"}'
    result = _strip_jsonc_comments(text)
    assert '/*' not in result
    assert '*/' not in result
    # Should parse as valid JSON
    parsed = json.loads(result)
    assert parsed["key"] == "value"


def test_strip_jsonc_preserves_strings():
    text = '{"url": "https://example.com/path // not a comment"}'
    result = _strip_jsonc_comments(text)
    parsed = json.loads(result)
    assert "// not a comment" in parsed["url"]


# ---- Deep merge ----

def test_deep_merge_basic():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 10, "e": 4}, "f": 5}
    result = _deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"]["c"] == 10
    assert result["b"]["d"] == 3
    assert result["b"]["e"] == 4
    assert result["f"] == 5


def test_deep_merge_instructions_concatenated():
    base = {"instructions": ["a", "b"]}
    override = {"instructions": ["c"]}
    result = _deep_merge(base, override)
    assert result["instructions"] == ["a", "b", "c"]


def test_deep_merge_non_instructions_list_replaced():
    base = {"paths": ["a", "b"]}
    override = {"paths": ["c"]}
    result = _deep_merge(base, override)
    assert result["paths"] == ["c"]


# ---- Config loading from file ----

def test_load_json_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"model": {"provider_id": "anthropic", "model_id": "claude-3"}}, f)
        f.flush()
        result = _load_json_file(f.name)
    os.unlink(f.name)
    assert result["model"]["provider_id"] == "anthropic"


def test_load_jsonc_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonc', delete=False) as f:
        f.write('{\n  // Default model\n  "model": {"provider_id": "openai", "model_id": "gpt-4o"}\n}')
        f.flush()
        result = _load_json_file(f.name)
    os.unlink(f.name)
    assert result["model"]["provider_id"] == "openai"


# ---- Config loading from directory ----

def test_load_config_from_project_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'tcode.json')
        with open(config_path, 'w') as f:
            json.dump({
                "model": {"provider_id": "anthropic", "model_id": "claude-sonnet"},
                "provider": {
                    "anthropic": {"api_key": "sk-test-123"}
                },
            }, f)
        cfg = load_config(tmpdir)
        assert cfg.model.provider_id == "anthropic"
        assert cfg.model.model_id == "claude-sonnet"
        assert cfg.provider["anthropic"].api_key == "sk-test-123"


def test_load_config_defaults_when_no_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(tmpdir)
        assert cfg.model.provider_id == "litellm"
        assert cfg.model.model_id == "claude-haiku-4-5-20251001"
        assert cfg.provider == {}


# ---- Env overrides ----

def test_env_overrides_provider():
    os.environ['TCODE_PROVIDER_OPENAI_API_KEY'] = 'test-key-123'
    try:
        overrides = _env_overrides()
        assert overrides['provider']['openai']['api_key'] == 'test-key-123'
    finally:
        del os.environ['TCODE_PROVIDER_OPENAI_API_KEY']


def test_env_overrides_model():
    os.environ['TCODE_MODEL_PROVIDER'] = 'anthropic'
    os.environ['TCODE_MODEL_ID'] = 'claude-3-opus'
    try:
        overrides = _env_overrides()
        assert overrides['model']['provider_id'] == 'anthropic'
        assert overrides['model']['model_id'] == 'claude-3-opus'
    finally:
        del os.environ['TCODE_MODEL_PROVIDER']
        del os.environ['TCODE_MODEL_ID']


# ---- Variable substitution ----

def test_substitute_env_var():
    os.environ['TEST_TCODE_VAR'] = 'hello'
    try:
        result = _substitute_vars("{env:TEST_TCODE_VAR}")
        assert result == "hello"
    finally:
        del os.environ['TEST_TCODE_VAR']


def test_substitute_file_var():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("file-content")
        f.flush()
        result = _substitute_vars(f"{'{'}file:{f.name}{'}'}")
    os.unlink(f.name)
    assert result == "file-content"


def test_substitute_nested():
    os.environ['TEST_TCODE_NESTED'] = 'nested-value'
    try:
        result = _substitute_vars({"key": "{env:TEST_TCODE_NESTED}", "list": ["{env:TEST_TCODE_NESTED}"]})
        assert result["key"] == "nested-value"
        assert result["list"][0] == "nested-value"
    finally:
        del os.environ['TEST_TCODE_NESTED']


# ---- Schema validation ----

def test_config_schema_full():
    raw = {
        "provider": {"openai": {"api_key": "sk-123", "base_url": "https://api.openai.com"}},
        "model": {"provider_id": "openai", "model_id": "gpt-4o"},
        "agent": {"custom": {"prompt": "Be helpful", "steps": 10}},
        "mcp": {"myserver": {"type": "http", "url": "http://localhost:8080"}},
        "command": {"deploy": {"template": "Deploy to $1", "description": "Deploy app"}},
        "skill": {"paths": ["/path/to/skills"], "urls": ["https://skills.example.com"]},
        "permission": {"shell": "ask"},
        "instructions": ["Always be concise"],
    }
    cfg = TcodeConfig.model_validate(raw)
    assert cfg.provider["openai"].api_key == "sk-123"
    assert cfg.agent["custom"].prompt == "Be helpful"
    assert cfg.mcp["myserver"].url == "http://localhost:8080"
    assert cfg.command["deploy"].template == "Deploy to $1"
    assert cfg.skill.paths == ["/path/to/skills"]
    assert cfg.instructions == ["Always be concise"]


def test_config_schema_empty():
    cfg = TcodeConfig.model_validate({})
    assert cfg.model.provider_id == "litellm"
    assert cfg.provider == {}


# ---- Singleton ----

def test_get_config_returns_same_instance():
    import tcode.config as config_mod
    config_mod._config = None
    with tempfile.TemporaryDirectory() as tmpdir:
        config_mod.set_project_dir(tmpdir)
        c1 = config_mod.get_config()
        c2 = config_mod.get_config()
        assert c1 is c2
    config_mod._config = None


def test_reload_config_invalidates_cache():
    import tcode.config as config_mod
    config_mod._config = None
    with tempfile.TemporaryDirectory() as tmpdir:
        config_mod.set_project_dir(tmpdir)
        c1 = config_mod.get_config()
        c2 = config_mod.reload_config()
        assert c1 is not c2
    config_mod._config = None
