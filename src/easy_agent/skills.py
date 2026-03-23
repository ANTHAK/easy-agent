from __future__ import annotations

import importlib.util
import inspect
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from easy_agent.models import RunContext, ToolSpec
from easy_agent.tools import ToolHandler, ToolRegistry


class SkillMetadata(BaseModel):
    name: str
    description: str
    entry_type: str
    hook: str | None = None
    command: list[str] = Field(default_factory=list)
    args_template: list[str] = Field(default_factory=list)
    env_passthrough: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": True}
    )


def _token_allowed(tokens: list[str], allowed_prefixes: list[list[str]]) -> bool:
    return any(tokens[: len(prefix)] == prefix for prefix in allowed_prefixes)


class SkillLoader:
    def __init__(self, skill_paths: list[Path], allowed_commands: list[list[str]]) -> None:
        self.skill_paths = skill_paths
        self.allowed_commands = allowed_commands

    def discover(self) -> list[tuple[SkillMetadata, Path]]:
        discovered: list[tuple[SkillMetadata, Path]] = []
        for root in self.skill_paths:
            if root.is_file():
                continue
            if (root / "skill.yaml").exists():
                candidates = [root]
            else:
                candidates = [path for path in root.iterdir() if path.is_dir()]
            for candidate in candidates:
                manifest_path = candidate / "skill.yaml"
                if not manifest_path.exists():
                    continue
                with manifest_path.open("r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or {}
                discovered.append((SkillMetadata.model_validate(payload), candidate))
        return discovered

    def register(self, registry: ToolRegistry) -> list[SkillMetadata]:
        registered: list[SkillMetadata] = []
        for metadata, base_path in self.discover():
            registry.register(
                ToolSpec(
                    name=metadata.name,
                    description=metadata.description,
                    input_schema=metadata.input_schema,
                ),
                self._make_handler(metadata, base_path),
            )
            registered.append(metadata)
        return registered

    def _make_handler(self, metadata: SkillMetadata, base_path: Path) -> ToolHandler:
        if metadata.entry_type == "python":
            return self._python_handler(metadata, base_path)
        if metadata.entry_type == "command":
            return self._command_handler(metadata, base_path)
        raise ValueError(f"Unsupported skill entry_type: {metadata.entry_type}")

    def _python_handler(self, metadata: SkillMetadata, base_path: Path) -> ToolHandler:
        if not metadata.hook:
            raise ValueError(f"Python skill '{metadata.name}' requires a hook")
        module_name, function_name = metadata.hook.split(":")
        module_path = base_path / module_name
        spec = importlib.util.spec_from_file_location(
            f"easy_agent_skill_{metadata.name}",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load skill module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        handler = getattr(module, function_name)

        async def _run(arguments: dict[str, Any], context: RunContext) -> Any:
            result = handler(arguments, context)
            if inspect.isawaitable(result):
                return await result
            return result

        return _run

    def _command_handler(self, metadata: SkillMetadata, base_path: Path) -> ToolHandler:
        if not metadata.command:
            raise ValueError(f"Command skill '{metadata.name}' requires a command")

        def _run(arguments: dict[str, Any], context: RunContext) -> Any:
            del context
            rendered_args = [token.format(**arguments) for token in metadata.args_template]
            tokens = metadata.command + rendered_args
            if not _token_allowed(tokens, self.allowed_commands):
                raise PermissionError(f"Command is not allowed by whitelist: {tokens}")
            result = subprocess.run(
                tokens,
                cwd=base_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            return {"stdout": result.stdout.strip(), "stderr": result.stderr.strip()}

        return _run
