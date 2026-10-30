import asyncio
from typing import Optional
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# 1. 移除 HuggingFaceEndpointEmbeddings 的导入，保留并使用 OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from app.conf.app_config import EmbeddingConfig, app_config


class EmbeddingClientManager:
    def __init__(self, config: EmbeddingConfig):
        # 懒加载：客户端在模块导入阶段先不立即创建，避免启动时就发起外部依赖连接
        # 2. 将类型注解改为 OpenAIEmbeddings
        self.client: Optional[OpenAIEmbeddings] = None
        # 保存 Embedding 服务配置，供 init() 时组装服务访问地址使用
        self.config = config

    def _get_url(self):
        # 当前项目通过 host + port 访问外部已启动的 Embedding 推理服务
        return f"http://{self.config.host}:{self.config.port}"

    def init(self):
        # 3. 使用 OpenAIEmbeddings 连接 TEI 服务
        self.client = OpenAIEmbeddings(
            base_url=f"{self._get_url()}/v1",       # TEI 服务的 OpenAI 兼容接口路径
            api_key="unused",                       # TEI 默认不需要认证，填任意非空字符串
            model="tei-model",                      # 占位符，TEI 服务端会忽略该参数
            check_embedding_ctx_length=False        # 关键：禁用本地 tiktoken 分词，发送原始文本给 TEI
        )


# 模块级单例，供其他模块按需复用同一个客户端管理器
embedding_client_manager = EmbeddingClientManager(app_config.embedding)


if __name__ == "__main__":
    # 本地调试入口：初始化客户端后执行一次最小化向量化调用
    embedding_client_manager.init()
    client = embedding_client_manager.client

    async def test():
        # 使用示例文本验证 Embedding 服务是否可正常响应
        text = "What is deep learning?"
        query_result = await client.aembed_query(text)
        # 只打印前 3 个维度，便于快速确认返回结果结构正确
        print(query_result[:3])

    # 运行调试测试
    asyncio.run(test())