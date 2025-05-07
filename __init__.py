import asyncio
import hashlib
from contextlib import AsyncExitStack, suppress
from typing import Any, Dict, List, Optional, cast

import anyio
import json5
import mcp
from mcp.client.sse import sse_client
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic import Field

from nekro_agent.api.core import logger
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.services.agent.creator import ContentSegment, OpenAIChatMessage
from nekro_agent.services.plugin.base import ConfigBase, NekroPlugin, SandboxMethodType

plugin = NekroPlugin(
    name="MCP 工具插件",
    module_name="mcp_tools",
    description="从 MCP SSE 服务获取工具并调用",
    version="0.1.0",
    author="Zaxpris_and_KroMiose",
    url="https://github.com/zxjwzn/nekro_plugin_mcp_connect",
)

@plugin.mount_config()
class MCPToolsConfig(ConfigBase):
    """MCP 服务配置"""
    MCP_CONFIG_JSON: str = Field(
        default="""
        {
            "servers": [
                {
                    "endpoint": "http://localhost:8080",
                    "enabled": false,
                    "description": "GitHub 仓库数据查询",//描述mcp服务内容,方便管理
                },
                {
                    "endpoint": "http://localhost:8081",
                    "enabled": false,
                    "description": "天气查询" //描述mcp服务内容,方便管理
                }
            ]
        }
        """,
        title="MCP 服务配置 (JSON格式)",
        description='使用 JSON 格式配置 MCP 服务，支持多个服务并指定其属性 <br><br>配置格式示例：<pre>{<br>  "servers": [<br>    {<br>      "name": "github",<br>      "endpoint": "http://localhost:8080",<br>      "enabled": true,<br>      "description": "GitHub 仓库数据查询"<br>    },<br>    {<br>      "name": "weather",<br>      "endpoint": "http://localhost:8081",<br>      "enabled": true,<br>      "description": "天气查询"<br>    }<br>  ]<br>}</pre>',
        json_schema_extra={"is_textarea": True},
    )

class MCPClient:
    def __init__(self, endpoint: str):
        self.name = "Unknown Server"
        self.endpoint = endpoint
        self.headers: Dict[str, str] = {}
        self.session: Optional[mcp.ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools: List[mcp.Tool] = []

    def set_auth_token(self, token: str) -> None:
        """
        设置一个认证令牌，用于在连接头中使用。

        Args:
            token: 认证令牌。
        """
        self.headers["Authorization"] = f"Bearer {token}"
        logger.info("认证令牌设置成功")

    async def connect(self):
        """连接到 MCP 服务器"""
        try:
            self.exit_stack = AsyncExitStack()
            # 使用 AsyncExitStack 管理 SSE 客户端上下文和 MCP 会话
            streams = await self.exit_stack.enter_async_context(
                    sse_client(self.endpoint, headers=self.headers),
                )
            read_stream, write_stream = streams
            self.session = await self.exit_stack.enter_async_context(
                    mcp.ClientSession(
                        read_stream, 
                        write_stream,
                    ),
                )
            init = await self.session.initialize()
            self.name = init.serverInfo.name
            logger.info(f"MCP 服务 [{self.name}] 连接成功")
        except Exception as e:
            logger.error(f"连接或初始化 MCP 会话失败: {e}", exc_info=True)
            if self.exit_stack:
                await self.exit_stack.aclose()
            self.exit_stack = None
            self.session = None
            raise

    async def load_tools(self):
        if not self.session:
            return
        result = await self.session.list_tools()
        self.tools = result.tools
        logger.info(f"MCP 服务 [{self.name}] 获取到 {len(self.tools)} 个工具")

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        if not self.session:
            raise RuntimeError(f"MCP 服务 [{self.name}] 未连接")
        try:
            return await self.session.call_tool(tool_name, params)
        except anyio.ClosedResourceError as e:
            logger.warning(f"底层流已关闭，重连 MCP 客户端 [{self.name}] 并重试: {e}")
            await self.reconnect()
            return await self.session.call_tool(tool_name, params)

    async def close(self):
        """关闭 MCP 会话及 SSE 上下文，忽略跨任务取消范围错误"""
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
                self.exit_stack = None
            self.session = None
        except RuntimeError as e:
            if "cancel scope" in str(e).lower():
                logger.warning(f"忽略 MCP 客户端 [{self.name}] 关闭时的跨任务取消范围错误: {e}")
            else:
                raise

    async def reconnect(self):
        """内部方法：关闭旧的 exit_stack 并重新连接 MCP 服务"""
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
        except RuntimeError as e:
            if "cancel scope" in str(e).lower():
                logger.warning(f"重连前关闭旧会话资源时忽略跨任务取消范围错误 [{self.name}]: {e}")
            else:
                logger.error(f"重连前关闭旧会话资源时发生意外的 RuntimeError [{self.name}]: {e}", exc_info=True)
        except Exception as e: # Catch other potential errors during aclose
            logger.warning(f"重连前关闭旧会话资源失败 [{self.name}]: {e}", exc_info=True)
        finally:
            # Ensure these are reset even if aclose fails partially
            self.exit_stack = None
            self.session = None

        # Proceed with reconnection
        await self.connect()  # This will create a new AsyncExitStack
        await self.load_tools()

# 全局 MCP 客户端实例字典，键为服务名
mcp_clients: Dict[str, MCPClient] = {}

# 当前配置哈希，用于变更检测
mcp_config_hash: Optional[str] = None

def _compute_config_hash(config_json: str) -> str:
    try:
        conf = json5.loads(config_json)
        norm = json5.dumps(conf, sort_keys=True)
    except Exception:
        norm = config_json
    return hashlib.md5(norm.encode("utf-8")).hexdigest()

@plugin.mount_init_method()
async def init_mcp_tools():
    """插件初始化时读取配置并初始化连接 MCP 服务"""
    global mcp_clients, mcp_config_hash
    config = plugin.get_config(MCPToolsConfig)
    try:
        config_dict = cast(Dict[str, Any], json5.loads(config.MCP_CONFIG_JSON))
        servers = config_dict.get("servers", [])
    except Exception as e:
        logger.error(f"解析 MCP 配置失败: {e}")
        return

    # 计算配置哈希，检测变更
    new_hash = _compute_config_hash(config.MCP_CONFIG_JSON)
    if mcp_config_hash == new_hash:
        logger.debug("MCP 配置未变化，跳过重新初始化")
        return
    # 配置变更：关闭并清空旧客户端
    for client in mcp_clients.values():
        await client.close()
    mcp_clients.clear()
    mcp_config_hash = new_hash
    for srv in servers:
        endpoint = srv.get("endpoint")
        enabled = srv.get("enabled", True)
        if not enabled:
            continue
        client = MCPClient(endpoint=endpoint)
        try:
            await client.connect()
            await client.load_tools()
            mcp_clients[client.name] = client
        except Exception as e:
            logger.error(f"初始化 MCP 客户端 [{client.name}] 失败: {e}")
    
    

@plugin.mount_prompt_inject_method("mcp_tools_prompt_inject")
async def mcp_tools_prompt_inject(_ctx: AgentCtx) -> str:
    """MCP 工具提示注入"""
    # 检测配置变更并重新初始化客户端
    await init_mcp_tools()
    # 根据已配置的 MCP 客户端生成工具及参数提示
    lines: List[str] = []
    if not mcp_clients:
        return ""
    lines.append("You can use the following tools to get information or perform actions by calling the mcp_call_tools function.")
    lines.append("The mcp_call_tools function accepts a list of tool calls, allowing for batch operations.")
    lines.append("Each tool call in the list should be a dictionary with 'server_name', 'tool_name', and 'params'.")

    for server_name, client in mcp_clients.items():
        lines.append(f"Service: {server_name}")
        if not client.tools:
            lines.append("  (No available tools)")
            continue
        for tool in client.tools:
            desc = tool.description or ""
            lines.append(f"  Tool: {tool.name} - {desc}")
            schema = tool.inputSchema or {}
            props = schema.get("properties", {})
            reqs = set(schema.get("required", []))
            if props:
                lines.append("    Parameters:")
                for prop, prop_schema in props.items():
                    ptype = prop_schema.get("type", "")
                    pdesc = prop_schema.get("description", "")
                    req_text = "Required" if prop in reqs else "Optional"
                    lines.append(f"      - {prop} ({ptype}, {req_text}) {pdesc}")
            else:
                lines.append("    No parameters")
    lines.append("Example for a single call: mcp_call_tools(tool_calls=[{'server_name': 'your_server', 'tool_name': 'your_tool', 'params': {'param1': 'value1'}}])")
    lines.append("Example for multiple calls: mcp_call_tools(tool_calls=[{'server_name': 'server1', 'tool_name': 'tool_a', 'params': {}}, {'server_name': 'server2', 'tool_name': 'tool_b', 'params': {'key': 'val'}}])")
    return "\n".join(lines)

@plugin.mount_sandbox_method(
    SandboxMethodType.MULTIMODAL_AGENT,
    name="调用 MCP 工具",
    description="通过 MCP 服务名、工具名和参数列表批量调用对应工具，并返回聚合的内容片段列表",
)
async def mcp_call_tools(_ctx: AgentCtx, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量调用 MCP 工具
    Args:
        tool_calls: 一个列表，其中每个元素是一个字典，包含:
            server_name: MCP 服务名
            tool_name: MCP 工具名
            params: 参数字典
    Returns:
        一个内容片段的列表 (List[Dict[str, Any]])。
        每个片段是一个字典，通常代表文本或图像内容。
        调用过程中的错误也会被格式化为文本片段并包含在列表中。

    不要尝试调用 MCP 工具插件 这个服务名, 这个服务名是插件本身, 而不是 MCP 服务
    正确的服务名应该是: github仓库搜索 天气查询 等等
    """
    # 检测配置变更并重新初始化客户端
    await init_mcp_tools()
    aggregated_segments: List[Dict[str, Any]] = []

    for call_spec in tool_calls:
        server_name = call_spec.get("server_name")
        tool_name = call_spec.get("tool_name")
        params = call_spec.get("params", {})

        if not server_name or not tool_name:
            error_text = f"MCP 调用跳过: 工具调用中缺少 'server_name' 或 'tool_name'。详情: {call_spec}"
            logger.warning(error_text)
            aggregated_segments.append(ContentSegment.text_content(error_text))
            continue

        client = mcp_clients.get(server_name)
        if not client:
            error_text = f"MCP 调用失败: 服务 [{server_name}] 未找到或未启用。工具: {tool_name}, 参数: {params}"
            logger.warning(error_text)
            aggregated_segments.append(ContentSegment.text_content(error_text))
            continue
        
        try:
            result: mcp.types.CallToolResult = await client.call_tool(tool_name, params)

            # 安全地访问 result.error
            if result.isError:
                error_text = f"MCP 工具 [{server_name}.{tool_name}] 调用失败。参数: {params}"
                logger.warning(error_text)
                aggregated_segments.append(ContentSegment.text_content(error_text))
                continue

            temp_msg = OpenAIChatMessage.create_empty("user")

            mcp_content_parts = result.content if result.content is not None else result.contents if result.contents is not None else []

            if not mcp_content_parts:
                logger.debug(f"MCP 工具 [{server_name}.{tool_name}] 执行成功，但未返回任何内容片段。参数: {params}")

            for part in mcp_content_parts:
                if isinstance(part, TextContent):
                    temp_msg.add(ContentSegment.text_content(part.text))
                elif isinstance(part, ImageContent):
                    mime_type = part.mimeType or "application/octet-stream"
                    data = part.data or ""
                    uri = f"data:{mime_type};base64,{data}"
                    temp_msg.add(ContentSegment.image_content(uri))
                elif isinstance(part, EmbeddedResource):
                    res = part.resource
                    if isinstance(res, TextResourceContents):
                        temp_msg.add(ContentSegment.text_content(res.text))
                    elif isinstance(res, BlobResourceContents):
                        mime_type = getattr(res, "mimeType", "application/octet-stream")
                        blob = getattr(res, "blob", "")
                        blob_uri = f"data:{mime_type};base64,{blob}"
                        temp_msg.add(ContentSegment.image_content(blob_uri))
                    else:
                        logger.warning(f"MCP 工具 [{server_name}.{tool_name}] 中遇到未处理的 EmbeddedResource 类型: {type(res)}")
                else:
                    logger.warning(f"MCP 工具 [{server_name}.{tool_name}] 中遇到未处理的 MCP ContentPart 类型: {type(part)}")
            
            aggregated_segments.extend(temp_msg.content)

        except Exception as e:
            error_text = f"调用或处理 MCP 工具 [{server_name}.{tool_name}] 时发生异常: {e}。参数: {params}"
            logger.exception(error_text)
            aggregated_segments.append(ContentSegment.text_content(error_text))
            
    return aggregated_segments

@plugin.mount_cleanup_method()
async def clean_up():
    """清理 MCP 工具插件资源"""
    global mcp_clients, mcp_config_hash, mcp_ping_tasks
    # 关闭所有客户端
    for client in mcp_clients.values():
        try:
            await client.close()
        except Exception as e:
            logger.error(f"关闭 MCP 客户端 [{client.name}] 失败: {e}")
    # 清空全局缓存和哈希
    mcp_clients.clear()
    mcp_config_hash = None