from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import anyio

from easy_agent.config import AgentConfig, AppConfig, GraphNodeConfig
from easy_agent.models import ChatMessage, NodeStatus, NodeType, RunContext, ToolSpec
from easy_agent.storage import SQLiteRunStore
from easy_agent.tools import ToolHandler, ToolRegistry


class AgentOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        model_client: Any,
        registry: ToolRegistry,
        store: SQLiteRunStore,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.registry = registry
        self.store = store
        self.agents: dict[str, AgentConfig] = config.agent_map

    def register_subagent_tools(self) -> None:
        for agent in self.config.graph.agents:
            for sub_agent_name in agent.sub_agents:
                tool_name = f"subagent__{sub_agent_name}"
                if self.registry.has(tool_name):
                    continue
                spec = self._subagent_spec(tool_name, sub_agent_name)
                runner = self._subagent_runner(sub_agent_name)
                self.registry.register(spec, runner)

    def _subagent_runner(self, target_name: str) -> ToolHandler:
        async def _run(arguments: dict[str, Any], context: RunContext) -> Any:
            prompt = str(arguments.get("prompt", ""))
            next_context = RunContext(
                run_id=context.run_id,
                workdir=context.workdir,
                node_id=context.node_id,
                shared_state=context.shared_state,
                depth=context.depth + 1,
            )
            return await self.run_agent(target_name, prompt, next_context)

        return _run

    @staticmethod
    def _subagent_spec(tool_name: str, agent_name: str) -> ToolSpec:
        return ToolSpec(
            name=tool_name,
            description=f"Delegate work to sub-agent '{agent_name}'.",
            input_schema={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        )

    async def run_agent(self, name: str, prompt: str, context: RunContext) -> Any:
        if context.depth > 6:
            raise RuntimeError("Maximum sub-agent depth exceeded")
        agent = self.agents[name]
        tool_names = agent.tools + [f"subagent__{item}" for item in agent.sub_agents]
        tool_specs = self.registry.list_specs(tool_names)
        messages = [
            ChatMessage(role="system", content=agent.system_prompt),
            ChatMessage(role="user", content=prompt),
        ]
        for iteration in range(agent.max_iterations):
            self.store.record_event(
                context.run_id,
                "agent_request",
                {"agent": name, "iteration": iteration + 1, "prompt": prompt},
            )
            response = await self.model_client.complete(messages, tool_specs)
            self.store.record_event(
                context.run_id,
                "agent_response",
                {
                    "agent": name,
                    "text": response.text,
                    "tool_calls": [item.model_dump() for item in response.tool_calls],
                },
            )
            if not response.tool_calls:
                return response.text
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )
            for tool_call in response.tool_calls:
                output = await self.registry.call(tool_call.name, tool_call.arguments, context)
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=str(output),
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                    )
                )
        raise RuntimeError(f"Agent '{name}' exceeded max_iterations")


class GraphScheduler:
    def __init__(
        self,
        config: AppConfig,
        registry: ToolRegistry,
        orchestrator: AgentOrchestrator,
        store: SQLiteRunStore,
        mcp_manager: Any,
    ) -> None:
        self.config = config
        self.registry = registry
        self.orchestrator = orchestrator
        self.store = store
        self.mcp_manager = mcp_manager

    async def run(self, input_text: str) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        self.store.create_run(run_id, self.config.graph.name, {"input": input_text})
        shared_state: dict[str, Any] = {"input": input_text}
        context = RunContext(
            run_id=run_id,
            workdir=Path.cwd(),
            node_id=None,
            shared_state=shared_state,
        )

        if self.config.graph.entrypoint in self.config.agent_map and not self.config.graph.nodes:
            output = await self.orchestrator.run_agent(
                self.config.graph.entrypoint,
                input_text,
                context,
            )
            self.store.finish_run(run_id, "succeeded", {"result": output})
            return {"run_id": run_id, "result": output}

        nodes = {node.id: node for node in self.config.graph.nodes}
        results: dict[str, Any] = {}
        remaining = set(nodes)
        while remaining:
            ready = [
                nodes[node_id]
                for node_id in remaining
                if all(dep in results for dep in nodes[node_id].deps)
            ]
            if not ready:
                self.store.finish_run(run_id, "failed", {"error": "cycle_or_missing_dependency"})
                raise RuntimeError("Graph contains unresolved dependencies or a cycle")
            for node in ready:
                output = await self._execute_node(node, results, context)
                results[node.id] = output
                shared_state[node.id] = output
                remaining.remove(node.id)

        final_output = results[self.config.graph.entrypoint]
        self.store.finish_run(run_id, "succeeded", {"result": final_output, "nodes": results})
        return {"run_id": run_id, "result": final_output, "nodes": results}

    async def _execute_node(
        self,
        node: GraphNodeConfig,
        results: dict[str, Any],
        parent_context: RunContext,
    ) -> Any:
        template_values = {"input": parent_context.shared_state["input"], **results}
        prompt = node.input_template.format(**template_values)
        node_context = RunContext(
            run_id=parent_context.run_id,
            workdir=parent_context.workdir,
            node_id=node.id,
            shared_state=parent_context.shared_state,
            depth=parent_context.depth,
        )
        last_error: Exception | None = None
        for attempt in range(node.retries + 1):
            self.store.record_node(
                parent_context.run_id,
                node.id,
                NodeStatus.RUNNING.value,
                attempt + 1,
                None,
                None,
            )
            try:
                with anyio.fail_after(node.timeout_seconds):
                    output = await self._dispatch_node(node, prompt, results, node_context)
                self.store.record_node(
                    parent_context.run_id,
                    node.id,
                    NodeStatus.SUCCEEDED.value,
                    attempt + 1,
                    output,
                    None,
                )
                return output
            except Exception as exc:
                last_error = exc
                self.store.record_node(
                    parent_context.run_id,
                    node.id,
                    NodeStatus.FAILED.value,
                    attempt + 1,
                    None,
                    str(exc),
                )
        if last_error is None:
            raise RuntimeError(f"Node '{node.id}' failed without an exception")
        raise last_error

    async def _dispatch_node(
        self,
        node: GraphNodeConfig,
        prompt: str,
        results: dict[str, Any],
        context: RunContext,
    ) -> Any:
        del results
        if node.type is NodeType.AGENT:
            if node.target is None:
                raise ValueError("Agent node requires target")
            return await self.orchestrator.run_agent(node.target, prompt, context)
        if node.type in (NodeType.TOOL, NodeType.SKILL):
            if node.target is None:
                raise ValueError("Tool/skill node requires target")
            payload = {"prompt": prompt, **node.arguments}
            return await self.registry.call(node.target, payload, context)
        if node.type is NodeType.MCP_TOOL:
            if node.target is None or "/" not in node.target:
                raise ValueError("mcp_tool target must be in the format 'server/tool'")
            server_name, tool_name = node.target.split("/", 1)
            payload = {"prompt": prompt, **node.arguments}
            return await self.mcp_manager.call_tool(server_name, tool_name, payload)
        if node.type is NodeType.JOIN:
            return {dep: context.shared_state[dep] for dep in node.deps}
        raise ValueError(f"Unsupported node type: {node.type}")
