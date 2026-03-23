# easy-agent

`easy-agent` 是一个偏白板、业务无关、可工程化扩展的 Agent 开发底座。首版目标是把运行时、协议适配、skills、MCP、multi-agent/subAgent 调度和可追踪性先打稳，让后续业务只是在这个组件上挂载配置和能力。

## 基线环境

- Python: `3.12.x`
- 虚拟环境: `uv venv --python 3.12`
- 安装依赖: `uv sync --dev`

## 快速开始

```powershell
uv venv --python 3.12
uv sync --dev
```

本地配置密钥:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
```

常用命令:

```powershell
uv run easy-agent doctor -c easy-agent.yml
uv run easy-agent skills list -c easy-agent.yml
uv run easy-agent run "用工具返回一句话" -c easy-agent.yml
```

## 设计要点

- `OpenAI`、`Anthropic`、`Gemini` 三类 tool-calling 协议统一映射到内部事件模型。
- skill 同时支持 `Python Hook` 和本地命令，两者都以工具方式注册。
- MCP 支持 `stdio` 和 `HTTP/SSE` 两种传输。
- 调度层支持 multi-agent / subAgent、任务图、重试、超时和 SQLite 轨迹持久化。

## 目录

- `src/easy_agent`: 核心运行时与 CLI。
- `examples/skills`: 示例 skills。
- `tests`: 单元与集成测试。
- `easy-agent.yml`: 声明式运行配置。
