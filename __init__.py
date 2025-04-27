import asyncio
import json
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field, model_validator

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger
from nekro_agent.services.plugin.base import ConfigBase, NekroPlugin, SandboxMethodType
from nekro_agent.tools.path_convertor import is_url_path

plugin = NekroPlugin(
    name="MCP 集成插件",
    module_name="mcp",
    description="提供 Model Context Protocol (MCP) 集成能力，允许用户配置多个 MCP 服务，使 AI 能够查询和调用外部数据和工具。",
    version="0.1.0",
    author="KroMiose",
    url="https://github.com/KroMiose/nekro-agent",
)


@plugin.mount_config()
class MCPConfig(ConfigBase):
    """MCP 配置"""

    MCP_CONFIG_JSON: str = Field(
        default="""{"servers": []}""",
        title="MCP 服务配置 (JSON格式)",
        description='使用 JSON 格式配置 MCP 服务，支持多个服务并指定其属性 <br><br>配置格式示例：<pre>{<br>  "servers": [<br>    {<br>      "name": "github",<br>      "endpoint": "http://localhost:8080",<br>      "enabled": true,<br>      "description": "GitHub 仓库数据查询",<br>      "auth_token": "your_token"<br>    },<br>    {<br>      "name": "weather",<br>      "endpoint": "http://localhost:8081",<br>      "enabled": true,<br>      "description": "天气查询"<br>    }<br>  ]<br>}</pre>',
        json_schema_extra={"is_textarea": True},
    )
    REQUEST_TIMEOUT: int = Field(
        default=10,
        title="请求超时时间（秒）",
        description="MCP 服务请求的超时时间，单位为秒",
    )
    MAX_RESPONSE_SIZE: int = Field(
        default=1024 * 1024 * 5,  # 默认 5MB
        title="最大响应大小（字节）",
        description="MCP 服务响应的最大大小，超过此大小的响应将被截断",
    )
    CACHE_EXPIRY: int = Field(
        default=0,  # 默认不缓存
        title="缓存过期时间（秒）",
        description="MCP 查询结果的缓存过期时间，0 表示不缓存",
    )
    HEALTH_CHECK_INTERVAL: int = Field(
        default=300,  # 默认 5 分钟
        title="健康检查间隔（秒）",
        description="MCP 服务健康检查的时间间隔，0 表示不进行健康检查",
    )
    DEFAULT_PROMPT_TEMPLATE: str = Field(
        default='**{service_name}** - {service_description}\n可通过以下方式使用此服务：\n- 查询: `mcp_query("{service_name}", "查询内容")`\n- 工具: `mcp_execute_tool("{service_name}", "工具名", 参数...)`\n- 资源: `mcp_get_resource("{service_name}", "资源路径")`',
        title="默认提示词模板",
        description="当服务未指定提示词模板时使用的模板，支持 {service_name} 和 {service_description} 占位符",
    )


# 获取配置
config: MCPConfig = plugin.get_config(MCPConfig)
store = plugin.store


class MCPServiceType(str, Enum):
    """MCP 服务类型"""

    QUERY = "query"  # 查询服务，提供信息查询功能
    TOOL = "tool"  # 工具服务，提供操作执行功能
    RESOURCE = "resource"  # 资源服务，提供资源访问功能
    HYBRID = "hybrid"  # 混合服务，提供多种功能


class MCPServerConfig(BaseModel):
    """MCP 服务配置"""

    name: str = Field(..., title="服务名称", description="唯一标识服务的名称")
    endpoint: str = Field(..., title="服务端点", description="MCP 服务的 HTTP 端点地址")
    enabled: bool = Field(True, title="是否启用", description="是否启用此 MCP 服务")
    description: str = Field("", title="服务描述", description="服务的简短描述")
    service_type: MCPServiceType = Field(
        MCPServiceType.HYBRID,
        title="服务类型",
        description="服务类型，决定了提供的功能",
    )
    prompt_template: str = Field(
        "",
        title="提示词模板",
        description="使用此服务的提示词模板，可包含 {service_name} 和 {service_description} 占位符",
    )
    resources: List[str] = Field(
        [],
        title="资源列表",
        description="服务提供的资源路径列表，如文件路径或 URL",
    )
    tools: List[str] = Field(
        [],
        title="工具列表",
        description="服务提供的工具列表，每个工具为一个命令字符串",
    )
    headers: Dict[str, str] = Field(
        {},
        title="请求头",
        description="发送到 MCP 服务的自定义请求头",
    )
    auth_token: str = Field(
        "",
        title="认证令牌",
        description="用于认证的令牌，将作为 Authorization Bearer 头发送",
        json_schema_extra={"is_secret": True},
    )

    @model_validator(mode="after")
    def validate_endpoint(self):
        """验证端点格式"""
        if not self.endpoint.startswith(("http://", "https://")):
            self.endpoint = f"http://{self.endpoint}"
        return self


class MCPServiceStatus(BaseModel):
    """MCP 服务状态"""

    name: str
    endpoint: str
    status: str  # "online", "offline", "error"
    last_check: float
    error_message: Optional[str] = None
    capabilities: List[str] = []
    version: Optional[str] = None


# 缓存 MCP 服务调用结果
mcp_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
# 服务状态缓存
service_status: Dict[str, MCPServiceStatus] = {}
# 解析后的服务配置
server_configs: List[MCPServerConfig] = []


def _parse_server_configs() -> List[MCPServerConfig]:
    """解析服务配置"""
    global server_configs

    try:
        # 解析JSON配置
        config_data = json.loads(config.MCP_CONFIG_JSON)
        if not isinstance(config_data, dict) or "servers" not in config_data:
            logger.error("MCP配置格式错误，必须包含 'servers' 数组")
            return []

        # 解析服务配置
        servers_data = config_data.get("servers", [])
        configs = []

        for server_data in servers_data:
            try:
                # 补充缺失的字段
                if "service_type" not in server_data:
                    server_data["service_type"] = "hybrid"

                server_config = MCPServerConfig(**server_data)
                configs.append(server_config)
            except Exception as e:
                logger.error(f"解析MCP服务配置失败: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"解析MCP配置JSON失败: {e}")
        return []
    except Exception as e:
        logger.error(f"处理MCP配置失败: {e}")
        return []
    else:
        server_configs = configs
        return configs


@plugin.mount_init_method()
async def init_plugin():
    """插件初始化"""
    # 解析服务配置
    servers = _parse_server_configs()
    logger.info(f"已加载 {len(servers)} 个 MCP 服务配置")

    # 初始化服务状态
    for server in servers:
        if server.enabled:
            await check_service_health(server.name)


async def check_service_health(service_name: str) -> MCPServiceStatus:
    """检查 MCP 服务健康状态

    Args:
        service_name: 服务名称

    Returns:
        MCPServiceStatus: 服务状态
    """
    global service_status

    # 确保服务配置已经解析
    if not server_configs:
        _parse_server_configs()

    # 查找服务配置
    server_config = next((s for s in server_configs if s.name == service_name), None)
    if not server_config:
        status = MCPServiceStatus(
            name=service_name,
            endpoint="unknown",
            status="error",
            last_check=time.time(),
            error_message=f"未找到名为 '{service_name}' 的 MCP 服务配置",
        )
        service_status[service_name] = status
        return status

    # 准备请求
    endpoint = server_config.endpoint
    if not endpoint.endswith("/health"):
        health_endpoint = f"{endpoint}/health"
    else:
        health_endpoint = endpoint

    try:
        headers = {}
        headers.update(server_config.headers)
        if server_config.auth_token:
            headers["Authorization"] = f"Bearer {server_config.auth_token}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                health_endpoint,
                headers=headers,
                timeout=5.0,  # 健康检查超时设为 5 秒
            )

            if response.status_code != 200:
                status = MCPServiceStatus(
                    name=service_name,
                    endpoint=endpoint,
                    status="error",
                    last_check=time.time(),
                    error_message=f"健康检查返回状态码 {response.status_code}: {response.text[:100]}",
                )
            else:
                try:
                    health_data = response.json()
                    status = MCPServiceStatus(
                        name=service_name,
                        endpoint=endpoint,
                        status="online",
                        last_check=time.time(),
                        capabilities=health_data.get("capabilities", []),
                        version=health_data.get("version"),
                    )
                except Exception as e:
                    status = MCPServiceStatus(
                        name=service_name,
                        endpoint=endpoint,
                        status="online",
                        last_check=time.time(),
                        error_message=f"无法解析健康检查响应: {e!s}",
                    )
    except Exception as e:
        status = MCPServiceStatus(
            name=service_name,
            endpoint=endpoint,
            status="offline",
            last_check=time.time(),
            error_message=f"健康检查失败: {e!s}",
        )

    # 更新服务状态
    service_status[service_name] = status
    return status


@plugin.mount_prompt_inject_method(name="mcp_prompt_inject")
async def mcp_prompt_inject(_ctx: AgentCtx) -> str:
    """MCP 服务提示词注入"""
    # 确保服务配置已经解析
    if not server_configs:
        _parse_server_configs()

    # 检查是否有已配置的服务
    active_servers = [s for s in server_configs if s.enabled]
    if not active_servers:
        return "你没有可用的 MCP 服务。你可以通过配置 MCP 服务来获取外部数据和工具。"

    # 更新服务健康状态
    if config.HEALTH_CHECK_INTERVAL > 0:
        for server in active_servers:
            # 检查是否需要更新健康状态
            if (
                server.name not in service_status
                or time.time() - service_status[server.name].last_check > config.HEALTH_CHECK_INTERVAL
            ):
                await check_service_health(server.name)

    # 构建提示词
    prompts = []
    prompts.append("## 可用的 MCP 服务")

    for server in active_servers:
        # 获取服务状态
        status_str = ""
        if server.name in service_status:
            status = service_status[server.name]
            status_emoji = "🟢" if status.status == "online" else "🔴" if status.status == "offline" else "⚠️"
            status_str = f"{status_emoji} "

        # 生成服务提示词
        template = server.prompt_template or config.DEFAULT_PROMPT_TEMPLATE
        prompt = template.format(
            service_name=server.name,
            service_description=server.description,
        )

        # 添加服务工具提示（如果有）
        if server.tools:
            tools_str = "可用工具:\n" + "\n".join([f"- `{tool}`" for tool in server.tools])
            prompt += f"\n\n{tools_str}"

        # 添加服务资源提示（如果有）
        if server.resources:
            resources_str = "可用资源:\n" + "\n".join([f"- `{Path(resource).name}`" for resource in server.resources])
            prompt += f"\n\n{resources_str}"

        prompts.append(f"{status_str}{prompt}")

    prompts.append("\n你可以通过以下方法与 MCP 服务交互：")
    prompts.append("1. `mcp_query(service_name, query)` - 向服务发送查询")
    prompts.append("2. `mcp_execute_tool(service_name, tool_name, *args)` - 执行服务工具")
    prompts.append("3. `mcp_get_resource(service_name, resource_path)` - 获取服务资源")
    prompts.append("4. `mcp_list_services()` - 列出所有可用服务及其状态")
    prompts.append(
        "\n**注意**：以上是可用的 MCP 方法，请仅使用这些方法与 MCP 服务交互，不要使用文档中提到的但未在此列出的其他 MCP 方法。",
    )

    return "\n\n".join(prompts)


async def _prepare_mcp_request(
    service_name: str,
    endpoint_suffix: str,
) -> tuple[Optional[MCPServerConfig], str, Dict[str, str]]:
    """准备 MCP 请求

    Args:
        service_name: 服务名称
        endpoint_suffix: 端点后缀，如 "/query"

    Returns:
        Tuple[Optional[MCPServerConfig], str, Dict[str, str]]: (服务配置, 完整端点URL, 请求头)
    """
    # 确保服务配置已经解析
    if not server_configs:
        _parse_server_configs()

    # 查找服务配置
    server_config = next((s for s in server_configs if s.name == service_name and s.enabled), None)
    if not server_config:
        return None, "", {}

    # 构建完整端点
    endpoint = server_config.endpoint
    if not endpoint.endswith(endpoint_suffix):
        endpoint = f"{endpoint}{endpoint_suffix}"

    # 准备请求头
    headers = {}
    headers.update(server_config.headers)
    if server_config.auth_token:
        headers["Authorization"] = f"Bearer {server_config.auth_token}"

    return server_config, endpoint, headers


async def _call_mcp_service(service_name: str, query: str) -> str:
    """调用 MCP 服务查询功能

    Args:
        service_name: 服务名称
        query: 查询内容

    Returns:
        str: 服务响应
    """
    server_config, endpoint, headers = await _prepare_mcp_request(service_name, "/query")
    if not server_config:
        return f"错误: 未找到名为 '{service_name}' 的可用 MCP 服务"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint,
                json={"query": query},
                headers=headers,
                timeout=config.REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                return f"错误: MCP 服务 '{service_name}' 返回状态码 {response.status_code}: {response.text[:100]}..."

            # 检查响应大小
            content_length = int(response.headers.get("Content-Length", "0"))
            if content_length > config.MAX_RESPONSE_SIZE:
                return f"错误: MCP 服务 '{service_name}' 响应大小 ({content_length} 字节) 超过限制 ({config.MAX_RESPONSE_SIZE} 字节)"

            result = response.json()
            return result.get("result", str(result))
    except httpx.TimeoutException:
        return f"错误: MCP 服务 '{service_name}' 请求超时 (>{config.REQUEST_TIMEOUT}秒)"
    except httpx.RequestError as e:
        return f"错误: MCP 服务连接失败: {e!s}"
    except Exception as e:
        logger.exception(f"MCP 服务调用异常: {e}")
        return f"错误: MCP 服务调用异常: {e!s}"


async def _execute_mcp_tool(service_name: str, tool_name: str, *args) -> str:
    """执行 MCP 服务工具

    Args:
        service_name: 服务名称
        tool_name: 工具名称
        args: 工具参数

    Returns:
        str: 执行结果
    """
    server_config, endpoint, headers = await _prepare_mcp_request(service_name, "/execute")
    if not server_config:
        return f"错误: 未找到名为 '{service_name}' 的可用 MCP 服务"

    # 检查工具是否存在
    if server_config.tools and tool_name not in server_config.tools:
        return f"错误: MCP 服务 '{service_name}' 不支持工具 '{tool_name}'"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint,
                json={
                    "tool": tool_name,
                    "args": list(args),
                },
                headers=headers,
                timeout=config.REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                return f"错误: MCP 服务 '{service_name}' 返回状态码 {response.status_code}: {response.text[:100]}..."

            result = response.json()
            return result.get("result", str(result))
    except httpx.TimeoutException:
        return f"错误: MCP 服务 '{service_name}' 请求超时 (>{config.REQUEST_TIMEOUT}秒)"
    except httpx.RequestError as e:
        return f"错误: MCP 服务连接失败: {e!s}"
    except Exception as e:
        logger.exception(f"MCP 工具执行异常: {e}")
        return f"错误: MCP 工具执行异常: {e!s}"


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="查询 MCP 服务",
    description="向 MCP 服务发送查询并获取结果",
)
async def mcp_query(_ctx: AgentCtx, service_name: str, query: str) -> str:
    """向 MCP 服务发送查询并获取结果

    用于向已配置的 MCP (Model Context Protocol) 服务发送查询，获取信息。

    Args:
        service_name (str): MCP 服务名称
        query (str): 查询内容

    Returns:
        str: 查询结果

    Example:
        ```python
        # 使用 Github MCP 服务查询仓库信息
        repo_info = mcp_query("github", "仓库 user/repo 的最新 commit 信息")

        # 使用 Weather MCP 服务查询天气
        weather = mcp_query("weather", "上海今天的天气如何？")

        # 使用 Knowledge MCP 服务查询知识
        knowledge = mcp_query("knowledge", "量子计算的基本原理是什么？")
        ```
    """
    # 获取缓存键
    cache_key = f"query:{query}"

    # 检查缓存
    chat_key = _ctx.from_chat_key
    now = time.time()
    if (
        chat_key in mcp_cache
        and service_name in mcp_cache[chat_key]
        and cache_key in mcp_cache[chat_key][service_name]
        and config.CACHE_EXPIRY > 0
    ):

        cache_entry = mcp_cache[chat_key][service_name][cache_key]
        if now - cache_entry["timestamp"] < config.CACHE_EXPIRY:
            return f"[MCP:{service_name}] 缓存结果 ({int(now - cache_entry['timestamp'])}秒前):\n{cache_entry['result']}"

    # 查询服务
    result = await _call_mcp_service(service_name, query)

    # 更新缓存
    if chat_key not in mcp_cache:
        mcp_cache[chat_key] = {}
    if service_name not in mcp_cache[chat_key]:
        mcp_cache[chat_key][service_name] = {}

    mcp_cache[chat_key][service_name][cache_key] = {
        "result": result,
        "timestamp": now,
    }

    return f"[MCP:{service_name}] 查询结果:\n{result}"


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="执行 MCP 服务工具",
    description="执行 MCP 服务提供的工具",
)
async def mcp_execute_tool(_ctx: AgentCtx, service_name: str, tool_name: str, *args) -> str:
    """执行 MCP 服务提供的工具

    用于执行已配置的 MCP 服务提供的工具功能。

    Args:
        service_name (str): MCP 服务名称
        tool_name (str): 工具名称
        *args: 工具参数

    Returns:
        str: 执行结果

    Example:
        ```python
        # 使用 Github MCP 服务创建 issue
        result = mcp_execute_tool("github", "create_issue", "user/repo", "Bug 报告", "发现了一个 Bug...")

        # 使用邮件 MCP 服务发送邮件
        result = mcp_execute_tool("email", "send", "recipient@example.com", "主题", "邮件内容...")

        # 使用文件 MCP 服务创建文件
        result = mcp_execute_tool("filesystem", "write", "/path/to/file.txt", "文件内容...")
        ```
    """
    # 执行工具
    result = await _execute_mcp_tool(service_name, tool_name, *args)

    return f"[MCP:{service_name}] 执行工具 '{tool_name}' 结果:\n{result}"


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="获取 MCP 资源",
    description="获取 MCP 服务提供的资源",
)
async def mcp_get_resource(_ctx: AgentCtx, service_name: str, resource_path: str) -> str:
    """获取 MCP 服务提供的资源

    获取已配置的 MCP 服务提供的资源，例如文件、文档等。

    Args:
        service_name (str): MCP 服务名称
        resource_path (str): 资源路径

    Returns:
        str: 资源内容或路径

    Example:
        ```python
        # 获取 Github MCP 服务提供的 README 文件
        readme = mcp_get_resource("github", "README.md")

        # 获取 Google Drive MCP 服务提供的文档
        document = mcp_get_resource("gdrive", "my_document.docx")
        ```
    """
    # 确保服务配置已经解析
    if not server_configs:
        _parse_server_configs()

    # 查找服务配置
    server_config = next((s for s in server_configs if s.name == service_name and s.enabled), None)
    if not server_config:
        return f"错误: 未找到名为 '{service_name}' 的可用 MCP 服务"

    # 检查资源是否存在
    if not server_config.resources:
        return f"错误: MCP 服务 '{service_name}' 未提供任何资源"

    # 查找资源
    matched_resources = [r for r in server_config.resources if r.endswith(resource_path) or Path(r).name == resource_path]
    if not matched_resources:
        return f"错误: 在 MCP 服务 '{service_name}' 中未找到资源 '{resource_path}'"

    # 获取资源
    resource = matched_resources[0]

    # 如果是 URL，返回 URL
    if is_url_path(resource):
        return resource

    # 如果是本地文件，返回文件路径
    path = Path(resource)
    if path.exists():
        return str(resource)

    return f"错误: 无法获取资源 '{resource_path}'"


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="列出 MCP 服务",
    description="列出所有可用的 MCP 服务及其状态",
)
async def mcp_list_services(_ctx: AgentCtx) -> str:
    """列出所有可用的 MCP 服务及其状态

    列出所有已配置的 MCP 服务及其当前状态，包括在线状态、版本和支持的功能。

    Returns:
        str: 服务列表字符串

    Example:
        ```python
        services = mcp_list_services()
        print(services)  # 显示所有可用的 MCP 服务
        ```
    """
    # 确保服务配置已经解析
    if not server_configs:
        _parse_server_configs()

    active_servers = [s for s in server_configs if s.enabled]
    if not active_servers:
        return "没有可用的 MCP 服务。"

    # 更新所有服务的健康状态
    for server in active_servers:
        if (
            server.name not in service_status
            or time.time() - service_status[server.name].last_check > config.HEALTH_CHECK_INTERVAL
        ):
            await check_service_health(server.name)

    # 构建服务列表
    service_list = []
    for server in active_servers:
        service_info = [f"服务名称: {server.name}"]
        service_info.append(f"描述: {server.description}")
        service_info.append(f"端点: {server.endpoint}")
        service_info.append(f"类型: {server.service_type}")

        # 添加状态信息
        if server.name in service_status:
            status = service_status[server.name]
            status_text = "在线" if status.status == "online" else "离线" if status.status == "offline" else "错误"
            service_info.append(f"状态: {status_text}")

            if status.version:
                service_info.append(f"版本: {status.version}")

            if status.capabilities:
                service_info.append(f"功能: {', '.join(status.capabilities)}")

            if status.error_message:
                service_info.append(f"错误: {status.error_message}")
        else:
            service_info.append("状态: 未知")

        # 添加工具和资源信息
        if server.tools:
            service_info.append(f"工具: {', '.join(server.tools)}")

        if server.resources:
            resources = [Path(r).name for r in server.resources]
            service_info.append(f"资源: {', '.join(resources)}")

        service_list.append("\n".join(service_info))

    return "\n\n".join(service_list)


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="检查 MCP 服务健康状态",
    description="检查指定 MCP 服务的健康状态",
)
async def mcp_check_health(_ctx: AgentCtx, service_name: str) -> str:
    """检查指定 MCP 服务的健康状态

    对指定的 MCP 服务执行健康检查，获取其状态、版本和支持的功能。

    Args:
        service_name (str): MCP 服务名称

    Returns:
        str: 健康检查结果

    Example:
        ```python
        health = mcp_check_health("github")
        print(health)  # 显示 Github MCP 服务的健康状态
        ```
    """
    # 执行健康检查
    status = await check_service_health(service_name)

    # 构建状态信息
    status_info = [f"服务名称: {status.name}"]
    status_info.append(f"端点: {status.endpoint}")

    status_text = "在线" if status.status == "online" else "离线" if status.status == "offline" else "错误"
    status_info.append(f"状态: {status_text}")

    if status.version:
        status_info.append(f"版本: {status.version}")

    if status.capabilities:
        status_info.append(f"功能: {', '.join(status.capabilities)}")

    if status.error_message:
        status_info.append(f"错误: {status.error_message}")

    return "\n".join(status_info)


@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件"""
    global mcp_cache, service_status, server_configs
    mcp_cache = {}
    service_status = {}
    server_configs = []
