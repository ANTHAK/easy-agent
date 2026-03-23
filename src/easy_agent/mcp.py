from __future__ import annotations

import asyncio
import json
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any, BinaryIO, cast

import httpx
from httpx_sse import aconnect_sse

from easy_agent.config import McpServerConfig
from easy_agent.models import ToolSpec


def _frame_message(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body


def _read_framed_sync(stream: BinaryIO) -> dict[str, Any]:
    headers = b""
    while b"\r\n\r\n" not in headers:
        chunk = stream.read(1)
        if not chunk:
            raise EOFError("MCP stream closed")
        headers += chunk
    raw_headers, _, remainder = headers.partition(b"\r\n\r\n")
    content_length = 0
    for line in raw_headers.decode("utf-8").split("\r\n"):
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
            break
    body = remainder
    while len(body) < content_length:
        chunk = stream.read(content_length - len(body))
        if not chunk:
            raise EOFError("MCP stream closed before message body completed")
        body += chunk
    return cast(dict[str, Any], json.loads(body.decode("utf-8")))


class BaseMcpClient(ABC):
    def __init__(self, config: McpServerConfig) -> None:
        self.config = config

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def list_tools(self) -> list[ToolSpec]: ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...

    @abstractmethod
    async def aclose(self) -> None: ...


class StdioMcpClient(BaseMcpClient):
    def __init__(self, config: McpServerConfig) -> None:
        super().__init__(config)
        self._process: subprocess.Popen[bytes] | None = None
        self._request_id = 0

    async def start(self) -> None:
        if not self.config.command:
            raise ValueError("stdio MCP transport requires a command")
        self._process = subprocess.Popen(
            self.config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
            env={**os.environ, **self.config.env},
        )

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("MCP stdio process is not running")
        self._request_id += 1
        payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        framed = _frame_message(payload)

        def _write() -> None:
            assert self._process is not None and self._process.stdin is not None
            self._process.stdin.write(framed)
            self._process.stdin.flush()

        await asyncio.to_thread(_write)
        stream = cast(BinaryIO, self._process.stdout)
        response = await asyncio.to_thread(_read_framed_sync, stream)
        return cast(dict[str, Any], response["result"])

    async def list_tools(self) -> list[ToolSpec]:
        result = await self._request("tools/list", {})
        return [
            ToolSpec(
                name=item["name"],
                description=item["description"],
                input_schema=item["inputSchema"],
            )
            for item in result["tools"]
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        return result["content"]

    async def aclose(self) -> None:
        if self._process is not None:
            self._process.terminate()
            await asyncio.to_thread(self._process.wait)


class HttpSseMcpClient(BaseMcpClient):
    def __init__(self, config: McpServerConfig) -> None:
        super().__init__(config)
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds)
        self.notifications: list[dict[str, Any]] = []
        self._sse_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.config.sse_url:
            self._sse_task = asyncio.create_task(self._consume_sse(self.config.sse_url))

    async def _consume_sse(self, url: str) -> None:
        async with aconnect_sse(self._client, "GET", url) as event_source:
            async for event in event_source.aiter_sse():
                if event.data:
                    self.notifications.append(cast(dict[str, Any], json.loads(event.data)))
                    return

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.config.rpc_url:
            raise ValueError("http_sse transport requires rpc_url")
        response = await self._client.post(
            self.config.rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        response.raise_for_status()
        payload = cast(dict[str, Any], response.json())
        return cast(dict[str, Any], payload["result"])

    async def list_tools(self) -> list[ToolSpec]:
        result = await self._rpc("tools/list", {})
        return [
            ToolSpec(
                name=item["name"],
                description=item["description"],
                input_schema=item["inputSchema"],
            )
            for item in result["tools"]
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        return result["content"]

    async def aclose(self) -> None:
        if self._sse_task is not None:
            self._sse_task.cancel()
        await self._client.aclose()


class McpClientManager:
    def __init__(self, configs: list[McpServerConfig]) -> None:
        self._clients: dict[str, BaseMcpClient] = {}
        for config in configs:
            if config.transport == "stdio":
                self._clients[config.name] = StdioMcpClient(config)
            elif config.transport == "http_sse":
                self._clients[config.name] = HttpSseMcpClient(config)
            else:
                raise ValueError(f"Unsupported MCP transport: {config.transport}")

    async def start(self) -> None:
        for client in self._clients.values():
            await client.start()

    async def list_servers(self) -> dict[str, list[ToolSpec]]:
        result: dict[str, list[ToolSpec]] = {}
        for name, client in self._clients.items():
            result[name] = await client.list_tools()
        return result

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        return await self._clients[server_name].call_tool(tool_name, arguments)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
