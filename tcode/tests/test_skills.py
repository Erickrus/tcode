"""Tests for tcode skill loader."""
from __future__ import annotations
import os
import tempfile
import pytest
from tcode.skills.loader import (
    _parse_frontmatter, _load_skill_md, _find_skill_mds,
    _discover_skill_dirs, discover_and_load_skills,
    load_skills, _load_py_skills, get_skill_dirs,
)
from tcode.tools import ToolRegistry


# ---- Frontmatter parsing ----

def test_parse_frontmatter_basic():
    text = "---\nname: my-skill\ndescription: A test skill\n---\nContent here"
    fm, body = _parse_frontmatter(text)
    assert fm["name"] == "my-skill"
    assert fm["description"] == "A test skill"
    assert body.strip() == "Content here"


def test_parse_frontmatter_no_frontmatter():
    text = "Just content without frontmatter"
    fm, body = _parse_frontmatter(text)
    assert fm == {}
    assert body == text


def test_parse_frontmatter_empty():
    text = "---\n---\nContent"
    fm, body = _parse_frontmatter(text)
    assert fm == {}
    assert body.strip() == "Content"


def test_parse_frontmatter_quoted_values():
    text = '---\nname: "quoted-name"\ndescription: \'single-quoted\'\n---\nbody'
    fm, body = _parse_frontmatter(text)
    assert fm["name"] == "quoted-name"
    assert fm["description"] == "single-quoted"


# ---- SKILL.md loading ----

def test_load_skill_md():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, dir=tempfile.gettempdir()) as f:
        f.write("---\nname: test-skill\ndescription: Test\n---\nDo something useful")
        f.flush()
        info = _load_skill_md(f.name)
    os.unlink(f.name)
    assert info is not None
    assert info["name"] == "test-skill"
    assert info["description"] == "Test"
    assert "Do something useful" in info["content"]


def test_load_skill_md_derives_name_from_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = os.path.join(tmpdir, 'my-skill')
        os.makedirs(skill_dir)
        skill_path = os.path.join(skill_dir, 'SKILL.md')
        with open(skill_path, 'w') as f:
            f.write("---\ndescription: No name in frontmatter\n---\nContent")
        info = _load_skill_md(skill_path)
        assert info["name"] == "my-skill"


def test_load_skill_md_missing_file():
    info = _load_skill_md("/nonexistent/SKILL.md")
    assert info is None


# ---- Find SKILL.md files ----

def test_find_skill_mds():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested SKILL.md files
        skill1 = os.path.join(tmpdir, 'skill1')
        skill2 = os.path.join(tmpdir, 'skill2', 'nested')
        os.makedirs(skill1)
        os.makedirs(skill2)
        with open(os.path.join(skill1, 'SKILL.md'), 'w') as f:
            f.write("---\nname: s1\n---\n")
        with open(os.path.join(skill2, 'SKILL.md'), 'w') as f:
            f.write("---\nname: s2\n---\n")
        # Also a non-skill file
        with open(os.path.join(tmpdir, 'README.md'), 'w') as f:
            f.write("Not a skill")

        results = _find_skill_mds(tmpdir)
        assert len(results) == 2


# ---- Discovery ----

def test_discover_and_load_skills():
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = os.path.join(tmpdir, '.tcode', 'skills', 'myskill')
        os.makedirs(skills_dir)
        with open(os.path.join(skills_dir, 'SKILL.md'), 'w') as f:
            f.write("---\nname: myskill\ndescription: My skill\n---\nDo the thing")

        result = discover_and_load_skills(project_dir=tmpdir)
        assert "myskill" in result
        assert result["myskill"]["description"] == "My skill"


def test_discover_extra_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_dir = os.path.join(tmpdir, 'extra_skills', 'askill')
        os.makedirs(extra_dir)
        with open(os.path.join(extra_dir, 'SKILL.md'), 'w') as f:
            f.write("---\nname: extra\n---\nExtra skill content")

        result = discover_and_load_skills(
            project_dir=tmpdir,
            extra_dirs=[os.path.join(tmpdir, 'extra_skills')],
        )
        assert "extra" in result


def test_duplicate_skill_warning(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two skills with same name in different dirs
        dir1 = os.path.join(tmpdir, '.tcode', 'skills', 'dup')
        dir2 = os.path.join(tmpdir, 'extra', 'dup')
        os.makedirs(dir1)
        os.makedirs(dir2)
        with open(os.path.join(dir1, 'SKILL.md'), 'w') as f:
            f.write("---\nname: dup\n---\nFirst")
        with open(os.path.join(dir2, 'SKILL.md'), 'w') as f:
            f.write("---\nname: dup\n---\nSecond")

        import logging
        with caplog.at_level(logging.WARNING):
            result = discover_and_load_skills(
                project_dir=tmpdir,
                extra_dirs=[os.path.join(tmpdir, 'extra')],
            )
        # First one wins
        assert result["dup"]["content"] == "First"
        assert "Duplicate" in caplog.text


# ---- Python module loading (legacy) ----

def test_load_py_skills():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = os.path.join(tmpdir, 'test_skill.py')
        with open(skill_path, 'w') as f:
            f.write("""
from tcode.tools import ToolInfo, ToolResult
from pydantic import BaseModel

class Params(BaseModel):
    msg: str

async def execute(args, ctx):
    return ToolResult(title="test", output=args.get('msg', ''), metadata={})

def register(registry):
    registry.register(ToolInfo(id="test_skill_tool", description="test", parameters=Params, execute=execute))
""")

        registry = ToolRegistry()
        _load_py_skills([tmpdir], registry)
        assert "test_skill_tool" in registry.list()


def test_load_skills_legacy_api():
    """Legacy load_skills() function works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = os.path.join(tmpdir, 'echo_skill.py')
        with open(skill_path, 'w') as f:
            f.write("""
from tcode.tools import ToolInfo, ToolResult
from pydantic import BaseModel
class P(BaseModel):
    text: str
async def execute(args, ctx):
    return ToolResult(title="echo", output=args.get('text',''), metadata={})
def register(registry):
    registry.register(ToolInfo(id="echo_skill", description="echo", parameters=P, execute=execute))
""")

        registry = ToolRegistry()
        load_skills([tmpdir], registry)
        assert "echo_skill" in registry.list()
