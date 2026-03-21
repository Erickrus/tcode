"""Skill loader for tcode.

Supports:
  - SKILL.md files with YAML frontmatter (name, description)
  - Python module skills (.py with register() function)
  - Multi-directory search: .tcode/skills/, .opencode/skills/, ~/.config/tcode/skills/
  - Remote skill URLs with local caching
  - Duplicate name warnings

Reference: opencode skill/skill.ts, skill/discovery.ts
"""
from __future__ import annotations
import importlib.util
import importlib
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)


# ---- SKILL.md frontmatter parsing ----

_FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)---\s*\n', re.DOTALL)


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Parse YAML-like frontmatter from a markdown file.

    Returns (frontmatter_dict, content_body).
    Simple key: value parsing (no nested YAML).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    fm_text = match.group(1)
    body = text[match.end():]

    fm: Dict[str, str] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            key, _, val = line.partition(':')
            fm[key.strip()] = val.strip().strip('"').strip("'")

    return fm, body


def _load_skill_md(path: str) -> Optional[Dict[str, Any]]:
    """Load a SKILL.md file and return skill info dict."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logger.warning(f"Failed to read skill file {path}: {e}")
        return None

    fm, body = _parse_frontmatter(text)
    name = fm.get('name', '')
    if not name:
        # Derive name from parent directory
        name = os.path.basename(os.path.dirname(path))
    if not name:
        name = os.path.splitext(os.path.basename(path))[0]

    return {
        'name': name,
        'description': fm.get('description', ''),
        'location': path,
        'content': body.strip(),
    }


# ---- Multi-directory skill discovery ----

def _discover_skill_dirs(project_dir: Optional[str] = None) -> List[str]:
    """Build the list of skill directories to search.

    Order (later overrides earlier for duplicate names):
      1. ~/.config/tcode/skills/
      2. .tcode/skills/ (in project dir)
      3. .opencode/skills/ (in project dir)
    """
    dirs = []
    home = os.path.expanduser('~')

    # Global skills
    global_skills = os.path.join(home, '.config', 'tcode', 'skills')
    if os.path.isdir(global_skills):
        dirs.append(global_skills)

    # Project skills
    if project_dir:
        for subdir in ('.tcode/skills', '.opencode/skills'):
            d = os.path.join(project_dir, subdir)
            if os.path.isdir(d):
                dirs.append(d)

    return dirs


def _find_skill_mds(directory: str) -> List[str]:
    """Find all SKILL.md files recursively in a directory."""
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.upper() == 'SKILL.MD':
                results.append(os.path.join(root, f))
    return results


# ---- Remote skill download ----

def _download_skill_index(base_url: str, cache_dir: str) -> List[str]:
    """Download skill index from a remote URL and cache files locally.

    Returns list of local directories containing SKILL.md files.
    """
    import urllib.request
    import urllib.error

    # Fetch index.json
    index_url = base_url.rstrip('/') + '/index.json'
    try:
        with urllib.request.urlopen(index_url, timeout=30) as resp:
            index = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        logger.warning(f"Failed to fetch skill index from {index_url}: {e}")
        return []

    skills = index.get('skills', [])
    if not isinstance(skills, list):
        return []

    result_dirs = []
    for entry in skills:
        name = entry.get('name', '')
        files = entry.get('files', [])
        if not name or not files:
            continue

        skill_dir = os.path.join(cache_dir, name)
        os.makedirs(skill_dir, exist_ok=True)

        for fname in files:
            local_path = os.path.join(skill_dir, fname)
            if os.path.isfile(local_path):
                continue  # Use cache
            file_url = f"{base_url.rstrip('/')}/{name}/{fname}"
            try:
                with urllib.request.urlopen(file_url, timeout=30) as resp:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, 'wb') as f:
                        f.write(resp.read())
            except Exception as e:
                logger.warning(f"Failed to download {file_url}: {e}")

        # Check if SKILL.md exists
        skill_md = os.path.join(skill_dir, 'SKILL.md')
        if os.path.isfile(skill_md):
            result_dirs.append(skill_dir)

    return result_dirs


# ---- Python module loading (preserved from original) ----

def _load_py_skills(skills_dirs: List[str], registry: ToolRegistry):
    """Load Python module skills (.py files with register() function)."""
    for d in skills_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.py'):
                continue
            path = os.path.join(d, fn)
            mod_name = f"skill_{os.path.splitext(fn)[0]}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if not spec:
                continue
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, 'register'):
                    mod.register(registry)
            except Exception:
                continue


# ---- Main API ----

def load_skills(skills_dirs: List[str], registry: ToolRegistry):
    """Legacy API: Load Python module skills from directories.

    Preserved for backward compatibility.
    """
    _load_py_skills(skills_dirs, registry)


def discover_and_load_skills(
    project_dir: Optional[str] = None,
    extra_dirs: Optional[List[str]] = None,
    remote_urls: Optional[List[str]] = None,
    registry: Optional[ToolRegistry] = None,
) -> Dict[str, Dict[str, Any]]:
    """Full skill discovery: SKILL.md + .py modules from multiple directories.

    Returns dict of {name: skill_info} for all discovered skills.
    Also loads .py module skills into the registry if provided.
    """
    all_skills: Dict[str, Dict[str, Any]] = {}
    skill_dirs: List[str] = []

    # 1. Built-in discovery directories
    discovered = _discover_skill_dirs(project_dir)
    skill_dirs.extend(discovered)

    # 2. Extra directories from config
    if extra_dirs:
        for d in extra_dirs:
            d = os.path.expanduser(d)
            if not os.path.isabs(d) and project_dir:
                d = os.path.join(project_dir, d)
            if os.path.isdir(d):
                skill_dirs.append(d)

    # 3. Remote URLs
    if remote_urls:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'tcode', 'skills')
        os.makedirs(cache_dir, exist_ok=True)
        for url in remote_urls:
            try:
                remote_dirs = _download_skill_index(url, cache_dir)
                skill_dirs.extend(remote_dirs)
            except Exception as e:
                logger.warning(f"Failed to load remote skills from {url}: {e}")

    # 4. Find and load SKILL.md files
    for d in skill_dirs:
        md_files = _find_skill_mds(d)
        for md_path in md_files:
            info = _load_skill_md(md_path)
            if not info:
                continue
            name = info['name']
            if name in all_skills:
                logger.warning(
                    f"Duplicate skill name '{name}': "
                    f"{all_skills[name]['location']} vs {info['location']} "
                    f"(keeping first)"
                )
                continue
            all_skills[name] = info

    # 5. Load .py module skills into registry
    if registry:
        _load_py_skills(skill_dirs, registry)

    return all_skills


def get_skill_dirs(project_dir: Optional[str] = None) -> List[str]:
    """Get all skill directories (for permission whitelisting)."""
    return _discover_skill_dirs(project_dir)
