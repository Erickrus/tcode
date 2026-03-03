from __future__ import annotations
import json
from typing import Dict, Any

# Minimal JSON schema-like validator for simple 'type' checks present in our schemas.
# This is NOT a full JSON Schema implementation; it's a small validator for MVP.

def validate_simple_schema(instance: Any, schema: Dict[str, Any]) -> (bool, str):
    # schema is expected to be {"type": "object", "properties": {k: {"type": "string"|"number"}}}
    stype = schema.get('type')
    if stype != 'object':
        return False, 'schema.type != object'
    props = schema.get('properties') or {}
    if not isinstance(instance, dict):
        return False, 'instance not object'
    for k, v in props.items():
        expected = v.get('type')
        if k not in instance:
            return False, f'missing property {k}'
        val = instance[k]
        if expected == 'string' and not isinstance(val, str):
            return False, f'property {k} not string'
        if expected == 'number' and not isinstance(val, (int, float)):
            return False, f'property {k} not number'
    return True, ''
