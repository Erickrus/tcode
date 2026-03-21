#!/usr/bin/env python3
"""Fake Weather MCP server for testing tcode MCP integration.

Provides fake weather data for Paris, Shanghai, and New York.

Run standalone:
    python test_weather_mcp.py                  # default port 9753
    python test_weather_mcp.py --port 8888

Then configure in tcode.json:
    {
      "mcp": {
        "weather": {
          "type": "http",
          "url": "http://localhost:9753",
          "enabled": true
        }
      }
    }
"""
from __future__ import annotations
import argparse
import json
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# ---- Fake weather data ----

WEATHER_DB = {
    "paris": {
        "city": "Paris",
        "country": "France",
        "temperature_c": 18,
        "temperature_f": 64,
        "condition": "Partly Cloudy",
        "humidity": 65,
        "wind_speed_kmh": 12,
        "wind_direction": "SW",
        "forecast": "Mild with occasional clouds. Light rain expected in the evening.",
    },
    "shanghai": {
        "city": "Shanghai",
        "country": "China",
        "temperature_c": 26,
        "temperature_f": 79,
        "condition": "Sunny",
        "humidity": 72,
        "wind_speed_kmh": 8,
        "wind_direction": "SE",
        "forecast": "Warm and sunny. High humidity throughout the day.",
    },
    "new york": {
        "city": "New York",
        "country": "United States",
        "temperature_c": 14,
        "temperature_f": 57,
        "condition": "Rainy",
        "humidity": 80,
        "wind_speed_kmh": 20,
        "wind_direction": "NE",
        "forecast": "Cool with steady rain. Temperatures dropping overnight.",
    },
}

# ---- Tool definitions (MCP format) ----

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a city. Supported cities: Paris, Shanghai, New York.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g. Paris, Shanghai, New York)",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "list_cities",
        "description": "List all cities with available weather data.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---- Tool execution ----

def execute_tool(name: str, args: dict) -> dict:
    if name == "get_weather":
        city = (args.get("city") or "").strip().lower()
        data = WEATHER_DB.get(city)
        if not data:
            available = ", ".join(d["city"] for d in WEATHER_DB.values())
            return {"text": f"Unknown city: '{args.get('city')}'. Available cities: {available}"}
        lines = [
            f"Weather for {data['city']}, {data['country']}:",
            f"  Temperature: {data['temperature_c']}°C / {data['temperature_f']}°F",
            f"  Condition:   {data['condition']}",
            f"  Humidity:    {data['humidity']}%",
            f"  Wind:        {data['wind_speed_kmh']} km/h {data['wind_direction']}",
            f"  Forecast:    {data['forecast']}",
        ]
        return {"text": "\n".join(lines)}

    elif name == "list_cities":
        cities = [d["city"] for d in WEATHER_DB.values()]
        return {"text": f"Available cities: {', '.join(cities)}"}

    return {"text": f"Unknown tool: {name}"}


# ---- HTTP endpoints matching tcode's expected MCP interface ----

async def health(request: Request):
    return JSONResponse({"status": "ok"})


async def list_tools(request: Request):
    return JSONResponse(TOOLS)


async def call_tool(request: Request):
    """Handle POST /tools/{name}/call — tcode's primary call pattern."""
    name = request.path_params["name"]
    body = await request.json()
    args = body.get("args", {})
    result = execute_tool(name, args)
    return JSONResponse(result)


async def call_tool_generic(request: Request):
    """Handle POST /call — tcode's fallback call pattern."""
    body = await request.json()
    name = body.get("tool", "")
    args = body.get("args", {})
    result = execute_tool(name, args)
    return JSONResponse(result)


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/tools", list_tools, methods=["GET"]),
        Route("/tools/{name}/call", call_tool, methods=["POST"]),
        Route("/call", call_tool_generic, methods=["POST"]),
    ],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake Weather MCP Server")
    parser.add_argument("--port", type=int, default=9753)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"Starting weather MCP server on http://{args.host}:{args.port}")
    print(f"Tools: {', '.join(t['name'] for t in TOOLS)}")
    print(f"Cities: {', '.join(d['city'] for d in WEATHER_DB.values())}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
