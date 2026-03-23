from __future__ import annotations

from pathlib import Path
from typing import Any

from easy_agent.config import AppConfig, load_config
from easy_agent.graph import AgentOrchestrator, GraphScheduler
from easy_agent.mcp import McpClientManager
from easy_agent.protocols import HttpModelClient
from easy_agent.skills import SkillLoader
from easy_agent.storage import SQLiteRunStore
from easy_agent.tools import ToolRegistry


class EasyAgentRuntime:
    def __init__(
        self,
        config: AppConfig,
        model_client: Any,
        registry: ToolRegistry,
        store: SQLiteRunStore,
        mcp_manager: McpClientManager,
        orchestrator: AgentOrchestrator,
        scheduler: GraphScheduler,
        skills: list[Any],
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.registry = registry
        self.store = store
        self.mcp_manager = mcp_manager
        self.orchestrator = orchestrator
        self.scheduler = scheduler
        self.skills = skills

    async def start(self) -> None:
        await self.mcp_manager.start()

    async def run(self, input_text: str) -> dict[str, Any]:
        return await self.scheduler.run(input_text)

    async def aclose(self) -> None:
        await self.mcp_manager.aclose()
        await self.model_client.aclose()


def build_runtime(config_path: str | Path) -> EasyAgentRuntime:
    config = load_config(config_path)
    registry = ToolRegistry()
    store = SQLiteRunStore(Path(config.storage.path), config.storage.database)
    skill_loader = SkillLoader([Path(item.path) for item in config.skills], config.security.allowed_commands)
    loaded_skills = skill_loader.register(registry)
    mcp_manager = McpClientManager(config.mcp)
    model_client = HttpModelClient(config.model)
    orchestrator = AgentOrchestrator(config, model_client, registry, store)
    orchestrator.register_subagent_tools()
    scheduler = GraphScheduler(config, registry, orchestrator, store, mcp_manager)
    return EasyAgentRuntime(config, model_client, registry, store, mcp_manager, orchestrator, scheduler, loaded_skills)
