import openai # OpenAI 官方 Python 库
from google import genai # Google Generative AI Python 库
from typing import List, Union, Optional # 类型注解
import os # 用于访问环境变量

# 从常量模块导入相关常量
from ..core.constants import (
    DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_GEMINI_EMBEDDING_MODEL,
    ENV_VAR_OPENAI_API_KEY, ENV_VAR_GEMINI_API_KEY, DEFAULT_OPENAI_BASE_URL,
    EMBEDDING_API_LOG_NAME
)
# （如果需要日志记录，可以取消注释以下行）
# from astrbot.core.log import LogManager
# logger = LogManager.GetLogger(log_name=EMBEDDING_API_LOG_NAME)


class OpenAIEmbeddingAPI:
    """
    封装了对 OpenAI 兼容的 Embedding 服务的调用。
    支持通过环境变量或直接参数指定 API 密钥和基础 URL。
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_EMBEDDING_MODEL, # 使用常量作为默认模型
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        初始化 OpenAIEmbeddingAPI 客户端。

        Args:
            model (str, optional): 使用的嵌入模型的名称。
                                   默认为 `DEFAULT_OPENAI_EMBEDDING_MODEL` ("text-embedding-3-small")。
            api_key (Optional[str], optional): OpenAI API 密钥。如果未提供，则尝试从环境变量 `ENV_VAR_OPENAI_API_KEY` ("OPENAI_API_KEY") 读取。
                                               默认为 `None`。
            base_url (Optional[str], optional): 自定义的 API 端点 URL (例如，用于兼容本地部署的 OpenAI API 或 Azure OpenAI)。
                                                如果未提供，则使用 OpenAI 官方的默认 URL (`DEFAULT_OPENAI_BASE_URL`)。默认为 `None`。

        Raises:
            ValueError: 如果未能找到 API 密钥（既未在参数中提供，也未在环境变量中设置）。
        """
        self.model: str = model
        # ENV_VAR_OPENAI_API_KEY 是 "OPENAI_API_KEY"
        self.api_key: Optional[str] = api_key or os.getenv(ENV_VAR_OPENAI_API_KEY)
        # DEFAULT_OPENAI_BASE_URL 是 "https://api.openai.com/v1"
        self.base_url: str = base_url or DEFAULT_OPENAI_BASE_URL

        if not self.api_key: # 如果 API 密钥仍未获取到
            raise ValueError(f"必须提供 OpenAI API 密钥，或在环境变量 {ENV_VAR_OPENAI_API_KEY} 中设置。")

        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        # logger.info(f"OpenAIEmbeddingAPI 初始化完成。模型: {self.model}, Base URL: {self.base_url}")

    def test_connection(self) -> None:
        """
        测试与 OpenAI Embedding 服务的连接性。
        尝试使用一个简单的输入请求嵌入向量。

        Raises:
            ConnectionError: 如果连接测试失败（例如，API密钥无效、网络问题、模型不存在等）。
        """
        try:
            # 使用一个简短的中文词语进行测试
            _ = self.client.embeddings.create(input=["你好"], model=self.model)
            # logger.info("OpenAI Embedding 服务连接测试成功。")
        except Exception as e: # 捕获所有可能的 OpenAI API 异常
            # logger.error(f"OpenAI Embedding 服务连接测试失败: {e}", exc_info=True)
            raise ConnectionError(
                f"OpenAI Embedding 服务连接测试失败: {e}\n"
                f"请检查：1. API密钥是否正确且有足够额度；2. 网络连接是否通畅；3. 模型名称 '{self.model}' 是否可用；4. Base URL '{self.base_url}' 是否正确。"
            ) from e # 保留原始异常链

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        获取一个或多个文本的嵌入向量。

        Args:
            texts (Union[str, List[str]]): 需要获取嵌入的单个文本字符串，或文本字符串列表。

        Returns:
            List[List[float]]: 包含每个输入文本对应嵌入向量的列表。每个嵌入向量本身是一个浮点数列表。

        Raises:
            ConnectionError: 如果在获取嵌入过程中发生 API 调用错误。
            TypeError: 如果 `texts` 参数类型不正确。
        """
        try:
            # 如果输入是单个字符串，将其转换为包含单个元素的列表，因为 API 需要列表输入
            if isinstance(texts, str):
                input_texts: List[str] = [texts]
            elif isinstance(texts, list):
                input_texts = texts
            else:
                raise TypeError(f"输入参数 `texts` 必须是 str 或 List[str]，但收到了 {type(texts)}。")

            # 调用 OpenAI API 获取嵌入向量
            response = self.client.embeddings.create(input=input_texts, model=self.model)
            # 从响应中提取嵌入向量列表
            # response.data 是一个列表，每个元素包含一个 embedding 属性
            return [data.embedding for data in response.data]
        except Exception as e: # 捕获所有可能的 OpenAI API 异常
            # logger.error(f"从 OpenAI 获取文本嵌入时发生错误: {e}", exc_info=True)
            raise ConnectionError(
                f"从 OpenAI 获取文本嵌入时发生错误: {e}\n"
                f"请检查：1. API连接和配置；2. 输入文本是否符合模型要求；3. 模型 '{self.model}' 是否支持当前操作。"
            ) from e


class GeminiEmbeddingAPI:
    """
    封装了对 Google Gemini Embedding 服务的调用。
    支持通过环境变量或直接参数指定 API 密钥。
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_EMBEDDING_MODEL, # 使用常量作为默认模型
        api_key: Optional[str] = None
    ) -> None:
        """
        初始化 GeminiEmbeddingAPI 客户端。

        Args:
            model (str, optional): 使用的 Gemini 嵌入模型的名称。
                                   默认为 `DEFAULT_GEMINI_EMBEDDING_MODEL` ("gemini-embedding-exp-03-07")。
            api_key (Optional[str], optional): Google Gemini API 密钥。如果未提供，则尝试从环境变量 `ENV_VAR_GEMINI_API_KEY` ("GEMINI_API_KEY") 读取。
                                               默认为 `None`。
        Raises:
            ValueError: 如果未能找到 API 密钥。
        """
        self.model: str = model
        # ENV_VAR_GEMINI_API_KEY 是 "GEMINI_API_KEY"
        self.api_key: Optional[str] = api_key or os.getenv(ENV_VAR_GEMINI_API_KEY)

        if not self.api_key: # 如果 API 密钥仍未获取到
            raise ValueError(f"必须提供 Google Gemini API 密钥，或在环境变量 {ENV_VAR_GEMINI_API_KEY} 中设置。")

        # 初始化 Google Generative AI 客户端
        # 注意：google.generativeai.configure(api_key=...) 是全局配置方式，
        # 如果需要每个实例有独立配置，或避免全局状态，应查阅库是否有实例级配置。
        # 当前 genai.Client(api_key=...) 似乎是推荐的实例级配置方式。
        self.client = genai.GenerativeModel(model_name=self.model, client=genai.Client(api_key=self.api_key)) # type: ignore
        # logger.info(f"GeminiEmbeddingAPI 初始化完成。模型: {self.model}")


    def test_connection(self) -> None:
        """
        测试与 Google Gemini Embedding 服务的连接性。
        尝试使用一个简单的输入请求嵌入向量。

        Raises:
            ConnectionError: 如果连接测试失败（例如，API密钥无效、网络问题、模型配置错误等）。
        """
        try:
            # 使用一个简短的英文短语进行测试
            # 注意：Gemini API 可能对测试内容有特定要求或格式
            _ = self.client.embed_content(content="hello world") # type: ignore
            # logger.info("Google Gemini Embedding 服务连接测试成功。")
        except Exception as e: # 捕获所有可能的 Gemini API 异常
            # logger.error(f"Google Gemini Embedding 服务连接测试失败: {e}", exc_info=True)
            raise ConnectionError(
                f"Google Gemini Embedding 服务连接测试失败: {e}\n"
                f"请检查：1. API密钥是否正确；2. 网络连接是否通畅；3. 模型名称 '{self.model}' 是否配置正确且可用。"
            ) from e

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        获取一个或多个文本的嵌入向量。

        Args:
            texts (Union[str, List[str]]): 需要获取嵌入的单个文本字符串，或文本字符串列表。

        Returns:
            List[List[float]]: 包含每个输入文本对应嵌入向量的列表。每个嵌入向量本身是一个浮点数列表。

        Raises:
            ConnectionError: 如果在获取嵌入过程中发生 API 调用错误。
            TypeError: 如果 `texts` 参数类型不正确。
        """
        try:
            # Gemini API 的 embed_content 通常接受 List[str] 作为 contents
            if isinstance(texts, str):
                input_texts: List[str] = [texts]
            elif isinstance(texts, list):
                input_texts = texts
            else:
                raise TypeError(f"输入参数 `texts` 必须是 str 或 List[str]，但收到了 {type(texts)}。")

            # 调用 Gemini API 获取嵌入向量
            # response 的结构是 EmbedContentResponse，其中包含一个 embeddings 列表
            # 每个 embedding 是一个 ContentEmbedding 对象，其 .values 属性是向量列表
            response = self.client.embed_content(requests=[{'model': self.model, 'content': content} for content in input_texts]) # type: ignore
            
            # 确保 response.embeddings 是一个列表
            if not isinstance(response.get("embeddings"), list):
                 raise ValueError(f"Gemini API 返回的 embeddings 格式不正确，期望列表但得到 {type(response.get('embeddings'))}")

            embeddings: List[List[float]] = [embedding.get("values") for embedding in response.get("embeddings", [])]
            return embeddings

        except Exception as e: # 捕获所有可能的 Gemini API 异常
            # logger.error(f"从 Google Gemini 获取文本嵌入时发生错误: {e}", exc_info=True)
            raise ConnectionError(
                f"从 Google Gemini 获取文本嵌入时发生错误: {e}\n"
                f"请检查：1. API连接和配置；2. 输入文本是否符合模型要求；3. 模型 '{self.model}' 是否支持当前操作。"
            ) from e