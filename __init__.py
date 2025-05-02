import hashlib
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

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
    author="Zaxpris & KroMiose",
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
                    "name": "github",
                    "endpoint": "http://localhost:8080",
                    "enabled": false,
                    "description": "GitHub 仓库数据查询"
                },
                {
                    "name": "weather",
                    "endpoint": "http://localhost:8081",
                    "enabled": false,
                    "description": "天气查询"
                }
            ]
        }
        """,
        title="MCP 服务配置 (JSON格式)",
        description='使用 JSON 格式配置 MCP 服务，支持多个服务并指定其属性 <br><br>配置格式示例：<pre>{<br>  "servers": [<br>    {<br>      "name": "github",<br>      "endpoint": "http://localhost:8080",<br>      "enabled": true,<br>      "description": "GitHub 仓库数据查询"<br>    },<br>    {<br>      "name": "weather",<br>      "endpoint": "http://localhost:8081",<br>      "enabled": true,<br>      "description": "天气查询"<br>    }<br>  ]<br>}</pre>',
        json_schema_extra={"is_textarea": True},
    )

class MCPClient:
    def __init__(self, name: str, endpoint: str):
        self.name = name
        self.endpoint = endpoint
        self.session: Optional[mcp.ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools: List[mcp.Tool] = []

    async def connect(self):
        """连接到 MCP 服务器"""
        # 使用 AsyncExitStack 管理 SSE 客户端上下文和 MCP 会话
        self._streams_context = sse_client(url=self.endpoint)
        try:
            # 进入 SSE 客户端上下文
            streams = await self.exit_stack.enter_async_context(self._streams_context)
            # 进入 MCP 会话上下文
            self.session = await self.exit_stack.enter_async_context(mcp.ClientSession(*streams))
            await self.session.initialize()
            logger.info(f"MCP 服务 [{self.name}] 连接成功")
        except Exception as e:
            logger.error(f"MCP 服务 [{self.name}] 连接失败: {e}")
            # 清理已进入的上下文
            await self.exit_stack.aclose()
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
        return await self.session.call_tool(tool_name, params)

    async def close(self):
        """关闭 MCP 会话及 SSE 上下文，忽略跨任务取消范围错误"""
        try:
            await self.exit_stack.aclose()
        except RuntimeError as e:
            if "cancel scope" in str(e).lower():
                logger.warning(f"忽略 MCP 客户端 [{self.name}] 关闭时的跨任务取消范围错误: {e}")
            else:
                raise

# 全局 MCP 客户端实例字典，键为服务名
mcp_clients: Dict[str, MCPClient] = {}

# 当前配置哈希，用于变更检测
mcp_config_hash: Optional[str] = None

def _compute_config_hash(config_json: str) -> str:
    try:
        conf = json.loads(config_json)
        norm = json.dumps(conf, sort_keys=True)
    except Exception:
        norm = config_json
    return hashlib.md5(norm.encode("utf-8")).hexdigest()

@plugin.mount_init_method()
async def init_mcp_tools():
    """插件初始化时读取配置并初始化连接 MCP 服务"""
    global mcp_clients, mcp_config_hash
    config = plugin.get_config(MCPToolsConfig)
    try:
        config_dict = json.loads(config.MCP_CONFIG_JSON)
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
        name = srv.get("name")
        endpoint = srv.get("endpoint")
        enabled = srv.get("enabled", True)
        if not name or not endpoint or not enabled:
            continue
        client = MCPClient(name=name, endpoint=endpoint)
        try:
            await client.connect()
            await client.load_tools()
            mcp_clients[name] = client
        except Exception as e:
            logger.error(f"初始化 MCP 客户端 [{name}] 失败: {e}")

@plugin.mount_prompt_inject_method("mcp_tools_prompt_inject")
async def mcp_tools_prompt_inject(_ctx: AgentCtx) -> str:
    """MCP 工具提示注入"""
    # 检测配置变更并重新初始化客户端
    await init_mcp_tools()
    # 根据已配置的 MCP 客户端生成工具及参数提示
    lines: List[str] = []
    if not mcp_clients:
        return ""
    lines.append("You can use the following tools to get information or perform actions.")
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
    lines.append("Example: mcp_call_tools(server_name, tool_name, params)")
    final_prompt = "\n".join(lines)
    return final_prompt

@plugin.mount_sandbox_method(
    SandboxMethodType.MULTIMODAL_AGENT,
    name="调用 MCP 工具",
    description="通过 MCP 服务名、工具名和参数调用对应工具",
)
async def mcp_call_tools(_ctx: AgentCtx, server_name: str, tool_name: str, params: Dict[str, Any]) -> Any:
    """调用 MCP 工具
    Args:
        server_name: MCP 服务名
        tool_name: MCP 工具名
        params: 参数字典
    """
    # 检测配置变更并重新初始化客户端
    await init_mcp_tools()
    client = mcp_clients.get(server_name)
    if not client:
        raise ValueError(f"MCP 服务 [{server_name}] 未找到或未启用")
    try:
        result = await client.call_tool(tool_name, params)
        # 构造多模态消息，类似 emotion 插件
        msg = OpenAIChatMessage.create_empty("user")
        # CallToolResult may use 'content' or 'contents'
        items = getattr(result, "content", None) or getattr(result, "contents", [])
        for item in items:
            if isinstance(item, TextContent):
                msg.add(ContentSegment.text_content(item.text))
            elif isinstance(item, ImageContent):
                # 使用 data URI 嵌入图片
                uri = f"data:{item.mimeType};base64,{item.data}"
                msg.add(ContentSegment.image_content(uri))
            elif isinstance(item, EmbeddedResource):
                res = item.resource
                if isinstance(res, TextResourceContents):
                    msg.add(ContentSegment.text_content(res.text))
                elif isinstance(res, BlobResourceContents):
                    blob_uri = f"data:{getattr(res, 'mimeType', 'application/octet-stream')};base64,{res.blob}"
                    msg.add(ContentSegment.image_content(blob_uri))
        # 直接返回列表形式的 content，避免将所有文本合并为字符串导致后续 extend 出错
        return {"role": msg.role, "content": msg.content}  # noqa: TRY300
    except Exception as e:
        logger.error(f"调用 MCP 工具 [{server_name}.{tool_name}] 失败: {e}")
        raise

@plugin.mount_cleanup_method()
async def clean_up():
    """清理 MCP 工具插件资源"""
    global mcp_clients, mcp_config_hash
    # 关闭所有客户端
    for client in mcp_clients.values():
        try:
            await client.close()
        except Exception as e:
            logger.error(f"关闭 MCP 客户端 [{client.name}] 失败: {e}")
    # 清空全局缓存和哈希
    mcp_clients.clear()
    mcp_config_hash = None