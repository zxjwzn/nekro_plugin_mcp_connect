"""
示例脚本：使用 anyio.run 自动管理上下文，演示连接远程 MCP SSE 服务，获取工具列表并调用。
"""
import anyio
import mcp
from mcp.client.sse import sse_client
import json

async def main():
    # SSE 模式配置示例，填写你的 MCP 服务 URL
    url = "https://mcp.api-inference.modelscope.cn/sse/c4ed65c0d11148"

    # 使用 async with 确保 enter/exit 在同一协程
    async with sse_client(url=url) as (read_stream, write_stream):
        async with mcp.ClientSession(read_stream, write_stream) as session:
            # 初始化会话
            await session.initialize()

            # 获取并打印可用工具列表
            resp = await session.list_tools()
            tools = resp.tools
            print("Available tools:")
            for t in tools:
                print(f"- {t.name}")
                print("  输入参数 schema:")
                print(json.dumps(t.inputSchema, indent=2, ensure_ascii=False))

            # 如果有工具，调用第一个示例
            if tools:
                tool_name = tools[0].name
                args = {"url": "https://www.baidu.com"}  # 根据工具 schema 填入参数
                print(f"Calling tool: {tool_name} with args: {args}")
                result = await session.call_tool(tool_name, args)
                print(f"Result from {tool_name}:", result)

    print("MCP session closed.")

if __name__ == "__main__":
    # 使用 anyio 运行，避免 CancelScope 跨 Task 问题
    anyio.run(main)