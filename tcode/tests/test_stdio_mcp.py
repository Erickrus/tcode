"""Tests for stdio MCP transport integration."""
import asyncio
import json
import os
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tcode.config import McpServerConfig, load_config
from tcode.mcp_transports import StdioTransport
from tcode.mcp import MCPManager

WEATHER_SCRIPT = os.path.join(os.path.dirname(__file__), '..', '..', 'weather', 'weather.py')


# ---- Config tests ----

def test_mcp_config_command_string_with_args():
    """Config accepts command as string + args as list (desired format)."""
    cfg = McpServerConfig(command='uv', args=['--directory', '/path', 'run', 'server.py'])
    assert cfg.type == 'local'
    assert cfg.command == 'uv'
    assert cfg.args == ['--directory', '/path', 'run', 'server.py']


def test_mcp_config_command_list():
    """Config accepts command as list (backward compatible)."""
    cfg = McpServerConfig(command=['uv', '--directory', '/path', 'run', 'server.py'])
    assert cfg.type == 'local'
    assert cfg.command == ['uv', '--directory', '/path', 'run', 'server.py']


def test_mcp_config_url_only():
    """Config with url only defaults to http type."""
    cfg = McpServerConfig(url='http://localhost:9753')
    assert cfg.type == 'http'
    assert cfg.command is None


def test_mcp_config_auto_detect_type():
    """Type is auto-detected as 'local' when command is present."""
    cfg = McpServerConfig(command='python', args=['server.py'])
    assert cfg.type == 'local'


def test_mcp_config_from_json():
    """Config parses from the desired tcode.json format."""
    raw = json.loads('{"command": "python", "args": ["/path/to/weather.py"]}')
    cfg = McpServerConfig(**raw)
    assert cfg.type == 'local'
    assert cfg.command == 'python'
    assert cfg.args == ['/path/to/weather.py']


def test_load_config_with_mcp_command():
    """load_config correctly parses MCP command-style config from tcode.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tcode_json = {
            'mcp': {
                'weather': {
                    'command': 'python',
                    'args': ['/content/weather/weather.py']
                }
            }
        }
        with open(os.path.join(tmpdir, 'tcode.json'), 'w') as f:
            json.dump(tcode_json, f)

        cfg = load_config(tmpdir)
        assert 'weather' in cfg.mcp
        weather = cfg.mcp['weather']
        assert weather.type == 'local'
        assert weather.command == 'python'
        assert weather.args == ['/content/weather/weather.py']


# ---- StdioTransport tests ----

@pytest.mark.asyncio
async def test_stdio_transport_connect():
    """StdioTransport connects to a stdio MCP server."""
    transport = StdioTransport('python', args=[WEATHER_SCRIPT])
    await transport.connect()
    assert transport._proc is not None
    await transport.close()


@pytest.mark.asyncio
async def test_stdio_transport_list_tools():
    """StdioTransport lists tools from a stdio MCP server."""
    transport = StdioTransport('python', args=[WEATHER_SCRIPT])
    await transport.connect()
    tools = await transport.list_tools()
    assert len(tools) == 2
    names = [t['name'] for t in tools]
    assert 'get_weather' in names
    assert 'list_cities' in names
    await transport.close()


@pytest.mark.asyncio
async def test_stdio_transport_call_tool():
    """StdioTransport calls tools on a stdio MCP server."""
    transport = StdioTransport('python', args=[WEATHER_SCRIPT])
    await transport.connect()

    chunks = []
    async for chunk in transport.call_tool('get_weather', {'city': 'Paris'}):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert 'Paris' in chunks[0]['text']
    assert '18°C' in chunks[0]['text']
    await transport.close()


@pytest.mark.asyncio
async def test_stdio_transport_call_list_cities():
    """StdioTransport calls list_cities tool."""
    transport = StdioTransport('python', args=[WEATHER_SCRIPT])
    await transport.connect()

    chunks = []
    async for chunk in transport.call_tool('list_cities', {}):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert 'Paris' in chunks[0]['text']
    assert 'Shanghai' in chunks[0]['text']
    assert 'New York' in chunks[0]['text']
    await transport.close()


# ---- MCPManager integration tests ----

@pytest.mark.asyncio
async def test_mcp_manager_stdio_add():
    """MCPManager.add() works with command-style (stdio) config."""
    mgr = MCPManager()
    config = {'command': 'python', 'args': [WEATHER_SCRIPT]}
    await mgr.add('weather', config)
    assert mgr.status['weather'] == 'connected'
    await mgr.remove('weather')


@pytest.mark.asyncio
async def test_mcp_manager_stdio_list_tools():
    """MCPManager lists tools from a stdio MCP server."""
    mgr = MCPManager()
    await mgr.add('weather', {'command': 'python', 'args': [WEATHER_SCRIPT]})
    tools = await mgr.list_tools('weather')
    assert 'get_weather' in tools
    assert 'list_cities' in tools
    await mgr.remove('weather')


@pytest.mark.asyncio
async def test_mcp_manager_stdio_call_tool():
    """MCPManager calls tools on a stdio MCP server."""
    mgr = MCPManager()
    await mgr.add('weather', {'command': 'python', 'args': [WEATHER_SCRIPT]})

    chunks = []
    async for chunk in mgr.call_tool('weather', 'get_weather', {'city': 'Shanghai'}):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert 'Shanghai' in chunks[0]['text']
    await mgr.remove('weather')


@pytest.mark.asyncio
async def test_mcp_manager_convert_tool():
    """MCPManager.convert_mcp_tool creates valid ToolInfo for stdio tools."""
    mgr = MCPManager()
    await mgr.add('weather', {'command': 'python', 'args': [WEATHER_SCRIPT]})

    client = mgr.clients['weather']
    raw_tools = await client.list_tools()
    for t in raw_tools:
        toolinfo = mgr.convert_mcp_tool('weather', t)
        assert toolinfo.id.startswith('mcp_weather_')
        assert toolinfo.description

    await mgr.remove('weather')


@pytest.mark.asyncio
async def test_mcp_manager_end_to_end_from_config():
    """Full end-to-end: load config -> add to MCPManager -> call tool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tcode_json = {
            'mcp': {
                'weather': {
                    'command': 'python',
                    'args': [WEATHER_SCRIPT]
                }
            }
        }
        with open(os.path.join(tmpdir, 'tcode.json'), 'w') as f:
            json.dump(tcode_json, f)

        cfg = load_config(tmpdir)
        mgr = MCPManager()

        for name, mcp_cfg in cfg.mcp.items():
            await mgr.add(name, mcp_cfg.model_dump())

        assert mgr.status['weather'] == 'connected'

        chunks = []
        async for chunk in mgr.call_tool('weather', 'get_weather', {'city': 'New York'}):
            chunks.append(chunk)

        assert 'New York' in chunks[0]['text']
        assert 'Rainy' in chunks[0]['text']

        await mgr.remove('weather')
