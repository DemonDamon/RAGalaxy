# Q1：【调整策略或提示词重试】怎么做可以提高性能

## 一、为什么要进行“调整策略或提示词重试”

1. 避免信息缺失：当一次性生成的摘要遗漏了重要信息，可通过提供新的提醒或要求，让模型在二次或多次推理中重新关注关键内容。  
2. 纠正语义偏差：若初次输出有歧义或偏离主题，通过改变提示语或加入更多上下文，可使模型输出更符合业务预期。  
3. 动态适配多场景：在同一流程里，不同文档或上下文环境下需要细调提示语，以兼容不同领域知识或风格。

---

## 二、可采用的调整策略

1. 增加“重点提示”（关键点列举）  
   - 当发现摘要缺少某些关键点时，可以在重试时明确指定需要保留的要点或关键词。  
   - 示例：“请重点提取文档中的以下关键信息：XXX、YYY，并保证不少于 N 字。”

2. 上下文对比  
   - 在重试前，把当前摘要与原文的关键段落做对比，自动识别缺失点，然后在提示语中添加“在上一版摘要中缺少以下内容，请重新总结时纳入：XXX”。  
   - 这样可以让模型聚焦处理缺失部分，减少总体重复。

3. 分层提示  
   - 当内容过多时，可以先让模型做分层总结，如“先列出文档的所有一级主题”，再逐个主题细化，使模型对整个文档的主题结构有全局认识后，再输出最终摘要。  
   - 如果某一层总结不满足需求，可在该层就地重试，而不必重做整份文档。

4. 使用示例或模板式提示  
   - 如果模型输出常常出现风格或结构上的偏差，可以提供一个“正确摘要示例”，并在提示语中说明：“请参照以下示例的格式，对文本做相同结构的摘要”。  
   - 有助于模型对输出格式和层级的把控。

5. 灵活控制长度  
   - 当摘要过长或过短时，可以在提示语中增加明确字数或段落数限制，比如“请将摘要限制在 500 字以内，并确保包含以下 3 点信息”。  
   - 这样既能保证浓缩度，也能避免内容过度删减。

---

## 三、生产环境落地的实现思路

要在上线生产的流程节点增加“策略或提示词重试”，可按以下思路设计：

1. 提示语/策略配置化  
   - 将提示语或策略放在可配置的文件（如数据库或配置中心）中，便于动态修改，而无需频繁改代码。  
   - 不同业务场景（如财务、法律、技术文档等）对应不同的提示模板，统一管理。

2. 重试条件定义  
   - 预先定义什么情况下需要进行重试（如：摘要与原文相似度过低、缺失必备关键词、摘要长度不达标等）。  
   - 通过自然语言处理、关键字匹配或人工审核结果来判定“是否触发重试”。

3. 重试提示语生成  
   - 在检测到需要重试后，可以自动拼接新的提示语，例如：  
     - “上一次结果中缺少以下关键信息：{missing_points}，请重新进行摘要。”  
     - “上一次结果过长，请将摘要控制在 {max_length} 字以内。”  
   - 同时保留原始题干和必要的上下文，避免模型完全丢失此前的信息。

4. 轮次限制/回退策略  
   - 为避免无限循环重试，可设置最大重试次数；超过次数后进入人工审核或日志记录队列。  
   - 确保线上流程不会被不合格的摘要卡住，造成生产级阻塞。

5. 日志与监控  
   - 为重试过程做埋点与日志记录，如：重试次数、重试原因、调优后的提示语、最终结果等。  
   - 事后可通过分析日志持续优化提示策略。

---

## 四、示例：用代码演示“提示词重试”逻辑

下面示例展示如何在已有摘要服务里，对“关键信息缺失”场景进行简单的提示词重试。假设我们有一个缺少的关键词列表，或者对摘要质量有自动检测机制。一旦检测到缺少必备关键词，就将这些关键词拼接到新的 Prompt 中做重试。

```python:summarization_retry.py
class SummarizationRetryHandler:
    def __init__(self, max_retries=2):
        self.max_retries = max_retries

    def check_missing_keywords(self, summary: str, key_points: list) -> list:
        """
        简易检测：若在摘要中找不到指定关键词，则返回缺失关键词列表
        """
        missing = []
        for kp in key_points:
            if kp not in summary:
                missing.append(kp)
        return missing

    def generate_retry_prompt(self, original_prompt: str, missing_points: list) -> str:
        """
        自动生成新的提示语，引导模型关注缺失要点
        """
        additional_instruction = f"请确保在摘要中包含以下关键信息: {', '.join(missing_points)}。"
        # 合并原始提示作上下文，再附加新指令
        retry_prompt = f"{original_prompt}\n{additional_instruction}"
        return retry_prompt

    def summarize_with_retry(self, text: str, key_points: list, summarize_func, original_prompt: str) -> str:
        """
        主函数：进行摘要生成并检查是否缺失关键词，不满足就重试。
        summarize_func: 传入实际的模型调用函数
        original_prompt: 第一次调用时用的提示语上下文
        """
        attempts = 0
        current_prompt = original_prompt
        final_summary = ""

        while attempts <= self.max_retries:
            # 1. 使用当前 prompt 调用模型生成摘要
            final_summary = summarize_func(text, current_prompt)

            # 2. 检查是否有缺失关键词
            missing = self.check_missing_keywords(final_summary, key_points)
            if len(missing) == 0:
                # 摘要合格，结束
                break
            else:
                # 如果还有尝试次数，则修改提示重试
                attempts += 1
                if attempts <= self.max_retries:
                    current_prompt = self.generate_retry_prompt(current_prompt, missing)
                else:
                    # 达到最大重试次数，直接返回最新的结果
                    break

        return final_summary
```

1. check_missing_keywords: 简易的规则检测，用于识别摘要中是否包含指定关键字。  
2. generate_retry_prompt: 将缺失的关键词作为强制性要求，以提示模型二次输出时必须包括这些内容。  
3. summarize_with_retry:  
   - 首次根据 original_prompt 生成摘要。  
   - 检测摘要是否缺少关键词；若缺失则自动调改 prompt，直到满足要求或达到最大重试次数。  
   - 若仍然不满足，就走回退逻辑或人工审核。

---

## 五、总结

在生产流程中加入“调整策略或提示词重试”节点，实质就是把 Prompt Engineering 的方法融入到自动化管线上，确保在摘要出现信息遗漏、话题偏移或其它不达标情况时，有可控的修正与补救措施。实践中应注意以下几点：

1. 重试次数要有限制，避免死循环。  
2. 提示语配置化，便于在线热更新和多业务场景的灵活切换。  
3. 根据业务需求灵活定义“合格”标准和校验方式（关键字、人工审核、主题对比等）。  
4. 监控和日志记录是提升模型性能与 Prompt 有效性的重要依据。  

通过上述机制，可在大部分情况下保证摘要质量，一旦遇到极端场景，还能有“回退到人工或后备方案”的兜底逻辑，从而使上线生产流程更具鲁棒性与可控性。

---

以下是一些在多分片摘要合并/二次汇总场景下，可能提升性能或效率的方向，涵盖逻辑层面的策略选择与工程层面的优化手段。由于每个项目背景和研发环境并不相同，以下方案仅供参考，可以根据具体业务需求和限制进行灵活调整。

---

# Q2：【多分片摘要合并/二次汇总】可能涉及多次调用大模型，怎么做可以提高性能
## 一、逻辑路由控制（策略）层面的优化

1. 分段策略动态化  
   - 根据文档长度与模型的实际输入限制，动态计算切分大小，避免切分过于粗放或过于细小。  
   - 针对已有文档结构（自然段落、标题、索引等），优先在语义合理的位置进行切分，减少后续生成摘要的重复与浪费，提高模型处理的有效性。

2. 适度分层汇总  
   - 如果一次性将所有分片摘要拼接得过多，会消耗模型资源且处理时间长，可考虑分层汇总：  
     - 先对分片进行初步摘要（多个批次进行）  
     - 然后将这些初步摘要再次汇总，分多轮逐步“收缩”  
     - 直到结果在可接受的长度范围内，再做最终汇总  
   - 这种做法以“分而治之”的方式，让每次模型输入都保持在一个合理的大小上，减少单次调用时间。

3. 选择性抽取（基于提示或规则）  
   - 如果分片内容有明确的核心要素（如“文章结论”“关键事实”等），可以在调用模型生成摘要前，通过提示或模板，让模型倾向于“重点抽取”而非“全局通读”，从而减少阅读量和生成量。  
   - 对已有结构化线索（例如，各章节的标题或纲要）可要求模型只在其范围内进行摘要，而不必全文扫描。

4. 分批并行调用 + 局部汇总  
   - 对于长文档的多个分片，可以考虑并行地调用模型生成分片摘要，然后再做合并；一旦发现分片摘要异常或无效，可以重试对应分片，而不必整篇重试。  
   - 合并阶段也可以分步处理，以减少一次性汇总带来的负担。

5. 模型类型与精度需求  
   - 根据实际业务需要，可以引入“多模型”策略：  
     - 第一阶段分片摘要可能用相对轻量的模型（如 qwen2.5_7B 级别，或其他精简模型）进行快速初步压缩；  
     - 二次汇总或最终关键信息输出再使用高精度大模型。  
   - 这样可以显著减少调用大模型的次数及成本。

---

## 二、工程层面的优化手段

1. 并行或异步处理  
   - 通过批量把多个分片摘要生成请求并行化或者异步化，可以有效缩短整体时间。  
   - Python 中可使用多线程或协程（如 asyncio）做批处理。  
   - 注意大模型服务端本身也要支持并行请求，否则可能造成性能瓶颈或资源竞争。

2. 缓存（Cache）机制  
   - 如果同一批文档中存在重复段落或者相似度极高的分片，可考虑在调用模型前进行去重处理；对于重复内容，直接复用之前的摘要结果。  
   - 将生成的摘要或摘要向量结果做短期缓存（例如 Redis），减少相同或相似请求时的重复调用。

3. 模型服务负载均衡  
   - 如果大模型是自建服务，可部署多副本或使用集群，利用负载均衡器（如 Nginx、K8s Ingress）实现多路并发。  
   - 针对各副本设置合理的资源配额与监控，防止超载导致服务不可用或极度延迟。

4. Prompt / 请求大小限制与分片  
   - 明确每次请求的 Token 或字符上限，采用“尽量小块、批量多次”的调用方式，避免一次性超长请求使模型内存暴增。  
   - 如果模型端支持压缩或二次摘要相关的特殊接口（如 intermediate summary），尽量利用该功能，让服务端在内部就分步压缩，减少往返通信量。

5. 限流与熔断  
   - 当系统同时处理大量文件时，应在接口服务端或调用端对模型接口进行限流保护，避免瞬时高并发引发 OOM 或阻塞。  
   - 可以在“并行/异步处理”中设置并发数上限，或者引入队列系统（如 RabbitMQ/Kafka）分发处理任务。

6. 硬件加速与架构选择  
   - 针对大模型推理，可以选择 GPU、半精度推理（FP16/BF16/Quantization）或分布式推理等方式，根据实际需求和预算进行相应的硬件加速。  
   - 对特定场景可采用量化或蒸馏模型来降低推理成本。

---

## 三、部分示例思路整合

下图（Mermaid）示例了一个结合逻辑与工程层面优化后，可能的二次汇总/多次调用流程（包含并行分片、部分缓存、分批合并等），仅供参考：

```mermaid
flowchart LR
    A([开始]) --> B{文档是否超限?}
    B -->|否| C[直接单次调用模型摘要]
    B -->|是| D[分片]
    D --> E[并行请求-多分片摘要]
    E -->|并行完成| F[结果缓存/去重检查]
    F --> G{分片结果是否太多?}
    G -->|多| H[先分段合并/二次汇总(并发可控)]
    G -->|少| I[直接拼接一次性汇总]
    H --> J[最终汇总再调用模型(多轮归纳)]
    I --> J
    J --> K[完成 & 写入ES/数据库]
    K --> L([结束])
```

说明：
1. 针对超长文本进行分片 -> 并行地调用摘要接口（E）  
2. 对生成的多个摘要结果可能进行去重或缓存（F），避免重复计算。  
3. 当分片数过多时（G），优先按照一定批次进行合并（H->J），减小负载；否则可直接一次性拼接后再汇总（I->J）。  
4. 最后存储进数据库或搜索引擎（K）。  

---

## 四、总结

提高多次调用大模型场景的性能，通常可以从以下两大方面入手：

1) 逻辑策略层面：  
   - 动态分片、分批合并、分层摘要、重点抽取、缓存与去重等，最大化减少冗余计算。  
   - 可以视项目情况采用“多模型架构”，用小模型做初步压缩，大模型深挖关键点。

2) 工程实现层面：  
   - 并行/异步处理、模型服务负载均衡、硬件加速、限流熔断、缓存等，提高系统吞吐量和稳定性。  
   - 通过日志监控、性能指标采集（调用时长、QPS、成功率等），及时发现瓶颈并定位优化方向。

结合以上思路，就能在【多分片摘要合并/二次汇总】场景下，有效降低对大模型的重复或无效调用，提升集中处理的性能与稳定性，从而更好地满足生产环境的需求。

# Q3：【多分片摘要合并/二次汇总】要怎么做？给出完整的满足生产上线的python工程代码我看看

以下示例演示了一个适用于生产环境的多分片摘要合并（即“二次汇总”）的 Python 工程示例，包含从文本分片、分片摘要、二次汇总、向量化到存储的完整逻辑，并在关键步骤增加了一些生产可用的考虑（如日志、错误重试、配置管理等）。请根据实际业务需求进行相应的功能扩充或微调。

## 工程目录结构示意

假设你的项目目录如下（仅示例）：

├── my_summarizer  
│   ├── config.py  
│   ├── services  
│   │   ├── model_client.py  
│   │   └── summary_service.py  
│   └── main.py  
└── requirements.txt  

- config.py：存放配置项（如模型地址、ES 地址等）  
- model_client.py：封装与大模型（qwen2.5_72B_128k）的交互方法  
- summary_service.py：核心文档总结流程逻辑：分片、摘要、二次汇总、向量化、存储  
- main.py：项目入口，示例演示如何调用流程  

以下分段代码示例将分别展示各文件内容。

---

```python:my_summarizer/config.py
import os

class Config:
    """
    生产环境常见的配置管理方式，可用环境变量或配置文件管理。
    这里仅作示例使用。
    """
    # 模型接口地址 (示例)
    MODEL_API_ENDPOINT = os.getenv("MODEL_API_ENDPOINT", "http://localhost:8000/model/qwen2.5_72B_128k")
    
    # Elasticsearch 配置
    ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
    ES_INDEX = os.getenv("ES_INDEX", "doc_summary")

    # 其他可选配置项
    DEFAULT_CHUNK_SIZE = 5000
    MAX_RETRIES = 3
```

---

```python:my_summarizer/services/model_client.py
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class QwenModelClient:
    """
    与 qwen2.5_72B_128k 模型交互的客户端示例。
    实际生产中可根据模型服务的具体接口协议进行HTTP或gRPC调用。
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def summarize(self, text: str) -> str:
        """
        摘要示例。实际中应根据模型接口实现。
        """
        try:
            # 例示使用HTTP请求
            payload = {"text": text}
            response = requests.post(f"{self.endpoint}/summarize", json=payload)
            response.raise_for_status()
            data = response.json()
            if "summary" in data:
                return data["summary"]
            else:
                logger.warning("模型返回数据格式不包含 'summary' 字段, 使用默认降级处理。")
                return "模型摘要结果缺失"
        except Exception as e:
            logger.error(f"调用模型接口出错: {e}")
            # 这里可进行降级处理，或返回空字符串或异常
            return "模型接口异常"

    def embed(self, text: str) -> list:
        """
        提取向量表征示例。实际中应根据模型接口实现。
        """
        try:
            payload = {"text": text}
            response = requests.post(f"{self.endpoint}/embed", json=payload)
            response.raise_for_status()
            data = response.json()
            if "embedding" in data:
                return data["embedding"]
            else:
                logger.warning("模型返回数据格式不包含 'embedding' 字段, 使用默认降级处理。")
                return []
        except Exception as e:
            logger.error(f"调用向量接口出错: {e}")
            return []
```

---

```python:my_summarizer/services/summary_service.py
import logging
from elasticsearch import Elasticsearch, helpers
from typing import List

from my_summarizer.config import Config
from my_summarizer.services.model_client import QwenModelClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DocumentSummarizer:
    """
    核心的文档总结服务类。
    包含：分片 -> 分片摘要 -> 多分片合并(二次汇总) -> 向量化 -> ES存储 等逻辑。
    """

    def __init__(self, es_host: str = None, es_index: str = None):
        self.es_host = es_host or Config.ES_HOST
        self.es_index = es_index or Config.ES_INDEX
        self.es = Elasticsearch(self.es_host)
        self.model_client = QwenModelClient(Config.MODEL_API_ENDPOINT)
        self.chunk_size = Config.DEFAULT_CHUNK_SIZE
        self.max_retries = Config.MAX_RETRIES

        # 初始化ES索引（若不存在可在此处创建mapping）
        self.init_index()

    def init_index(self):
        """
        示例：检查并初始化ES索引。
        实际可以根据业务需要设计mapping，比如使用dense_vector类型等。
        """
        if not self.es.indices.exists(index=self.es_index):
            logger.info(f"索引[{self.es_index}]不存在，尝试创建...")
            body = {
                "mappings": {
                    "properties": {
                        "summary_text": {"type": "text"},
                        "summary_vector": {
                            "type": "dense_vector",
                            "dims": 768  # 示例：向量维度
                        },
                        "summary_length": {"type": "integer"},
                        "chunk_count": {"type": "integer"}
                    }
                }
            }
            self.es.indices.create(index=self.es_index, body=body)
            logger.info(f"索引[{self.es_index}]创建成功。")

    def split_document(self, text: str, chunk_size: int = None) -> List[str]:
        """
        将原文档按指定字符大小切分为多个片段。
        """
        size = chunk_size or self.chunk_size
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

    def summarize_chunk(self, chunk: str) -> str:
        """
        对单个分片进行摘要, 带简单的重试机制。
        """
        for attempt in range(self.max_retries):
            try:
                summary = self.model_client.summarize(chunk)
                if summary:
                    return summary
            except Exception as e:
                logger.error(f"分片摘要调用失败: {e}, 第{attempt+1}次重试中...")
        # 最终失败则给一个默认占位
        return "摘要失败"

    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        """
        逐个调用模型对每个分片进行摘要。
        """
        results = []
        for idx, chunk in enumerate(chunks):
            summary = self.summarize_chunk(chunk)
            logger.info(f"[分片 {idx+1}/{len(chunks)}] 摘要完成。")
            results.append(summary)
        return results

    def merge_and_summarize(self, summaries: List[str]) -> str:
        """
        多分片摘要合并（拼接）后，再次调用模型做二次汇总。
        若分片非常多且每个分片摘要又很长，可分批次做多轮汇总。
        """
        # 简化做法：直接拼接后一次调用；若分片多且摘要较长，可分段多轮归纳
        combined_text = "\n".join(summaries)

        # 若 combined_text 过长，仍需二次切分 -> 再次请求 -> 直到摘要可控
        # 这里给出一个简单的“多轮收缩”示例
        current_text = combined_text
        round_count = 0
        while len(current_text) > self.chunk_size * 2:
            round_count += 1
            logger.info(f"二次汇总: 第 {round_count} 轮对 combined_text 进行再次压缩...")
            # 切分成一两个分片，再合并生成更短的middle summary
            mid_chunks = self.split_document(current_text, chunk_size=self.chunk_size)
            mid_summaries = []
            for mid_chunk in mid_chunks:
                mid_summaries.append(self.summarize_chunk(mid_chunk))
            current_text = "\n".join(mid_summaries)

        # 当文本长度可控后，进行最终二次汇总
        final_summary = self.summarize_chunk(current_text)
        return final_summary

    def vectorize_text(self, text: str) -> List[float]:
        """
        调用嵌入接口，将文本转换为向量。
        """
        for attempt in range(self.max_retries):
            try:
                vector = self.model_client.embed(text)
                if vector:
                    return vector
            except Exception as e:
                logger.error(f"向量化失败: {e}, 第{attempt+1}次重试中...")
        # 若最终失败则返回空向量
        return []

    def save_to_es(self, doc_id: str, summary_text: str, summary_vector: List[float], chunk_count: int):
        """
        将最终的摘要及相关信息写入ES。
        """
        body = {
            "summary_text": summary_text,
            "summary_vector": summary_vector,
            "summary_length": len(summary_text),
            "chunk_count": chunk_count
        }
        # 也可使用 bulk 批量写，这里仅演示单条写入
        self.es.index(index=self.es_index, id=doc_id, body=body)
        logger.info(f"文档 {doc_id} 的摘要数据已成功写入 ES。")

    def summarize_document(self, doc_id: str, text: str) -> str:
        """
        主流程:
          1. 检查文本长度 & 分片
          2. 分片摘要
          3. 若多片则进行多分片摘要合并/二次汇总
          4. 向量化 & 存储在ES
        """
        # 1. 分片
        chunks = self.split_document(text)
        logger.info(f"文档 {doc_id} 分片数: {len(chunks)}")

        # 2. 分片摘要
        chunk_summaries = self.summarize_chunks(chunks)

        # 3. 若多片则二次汇总
        if len(chunks) > 1:
            final_summary = self.merge_and_summarize(chunk_summaries)
        else:
            final_summary = chunk_summaries[0]

        # 4. 向量化 & 写入ES
        summary_vector = self.vectorize_text(final_summary)
        self.save_to_es(doc_id, final_summary, summary_vector, len(chunks))

        return final_summary
```

---

```python:my_summarizer/main.py
import logging
from my_summarizer.services.summary_service import DocumentSummarizer

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    # 示例文档
    example_text = (
        "这是一个非常长的文档内容，假设内容有好几万字……"
        "（此处省略大量文本，演示分片、摘要、二次汇总等流程）"
    )

    summarizer = DocumentSummarizer()
    doc_id = "example-doc-001"
    final_summary = summarizer.summarize_document(doc_id, example_text)

    print("===== 最终文档摘要 =====")
    print(final_summary)
```
## 代码关键点说明

1. 多分片摘要合并/二次汇总 (merge_and_summarize)  
   - 初始思路：先把所有分片的小摘要串起来，然后直接再调用模型生成一个“全局总结”。  
   - 如果分片数量多且每个分片摘要也很长，合并后文本依旧超出模型最大输入限制，则需要进一步分批压缩。上例中使用“多轮收缩”的方法，每次把文本分割为较小段落进行中间摘要，然后继续合并，直到长度可控后再做最终请求。  
   - 需要根据实际文档长度和模型的 token 上限做动态控制，以确保不会超出模型可处理范围。

2. 分片策略  
   - 工程中演示的是简单的“固定大小切分”方案，如需更高质量，可考虑“按语义或段落分割”再做微调。  
   - 当文档极长（例如 10 万字）时，可以灵活多次切分，以减少单次调用的负担。

3. 错误重试与日志  
   - 在模型调用、向量化等环节都加了简单的重试机制，以应对网络抖动或临时异常。  
   - 生产环境中可配合服务层的熔断和限流措施，或记录更多维度的监控指标（如调用时长、错误种类）。

4. ES 写入  
   - 示例中使用了同步写入；大量文档处理时，可以考虑批量方式（bulk）或异步队列。  
   - 建议在生产系统里做好索引的版本管理（mapping 变更时）和备份。

5. 其他增强思路  
   - 人工审核环节：若领域对摘要质量要求极高，可在写入 ES 前增加人工/规则校验。  
   - 多层摘要结构：在二次汇总基础上，可进一步分章节、主题做更加结构化的摘要输出。  
   - 向量检索：在 ES 中启用向量检索（knn search），可让后续查询更加满足语义化需求。

# Q4：具体告诉我【增加“关键信息校验”或“与原文比对”的逻辑】这里怎么实现？
以下给出一个针对“关键信息校验”或“与原文比对”逻辑的示例思路与参考代码实现示例。核心思想是：  
1. 从原文中抽取一批被视作必不可少的关键元素（可基于关键词、实体识别或业务特定字段）。  
2. 用一定策略（字符串匹配、词频/短句匹配、向量相似度等）在模型摘要中进行搜索，判断是否“缺失”。  
3. 若缺失比例超出预先设定阈值，则视为信息损耗过高，可进行重新摘要或提示人工校对。  

下面先从逻辑层面做示例说明，再给出相关伪代码供参考。

---

## 一、实现逻辑思路

1. 抽取原文关键词 / 关键信息  
   - 先对原文做分词或实体识别，找出与业务最相关的一批核心词或短句（例如：产品型号、人物姓名、时间、地名、专业术语等）。  
   - 若应高度自定义抽取逻辑，则可结合领域词典或特殊标记（如正则匹配、标注系统）来获取必需信息。  

2. 对摘要做匹配校验  
   - 逐项检查抽取到的关键词 / 短语是否在摘要中出现，或是否与摘要中的词或其近义词匹配。如果缺失过多，说明摘要可能遗漏了过多重要内容。  
   - 也可借助向量相似度（embedding）来更加灵活地判断关键词是否在摘要中被“变体”表达，以减少只做字符串匹配导致的漏检。  

3. 评分并做决策  
   - 设定一个缺失率阈值（例如 20%），若被判定的关键信息缺失率高于此阈值，则认为摘要质量不合格。  
   - 如不合格，可以执行以下操作：  
     a) 调整提示语再次生成摘要，让模型优先包含关键点。  
     b) 将当前摘要与原文一起，传给人工审阅；或  
     c) 尝试其他更稳健的总结方法（例如多模态汇总、缩小分片规模、增加提示词等）。  

---

## 二、示例代码

下面以 Python 为例。示范代码包括：  
1. 一个用于把原文进行关键词抽取的伪实现。  
2. 一个用于关键词/短语在摘要中的匹配校验核心逻辑。  
3. 一个主流程方法，将其集成到原有的“分片摘要生成 + 二次汇总”流程中，产生“关键信息校验”后再决定是否需要重试或提示。

请注意，根据实际环境和业务需求，实际线上代码中往往会使用专业的分词工具（如 jieba、HanLP，或使用自有词典），并且会针对关键词的重要程度分级筛选。

```python:services/document_summarizer.py
import re
from typing import List, Dict

class DocumentSummarizer:
    def __init__(self, ...):
        # 原有初始化逻辑
        pass

    def extract_key_info(self, text: str) -> List[str]:
        """
        从原文中抽取关键信息（如关键词、实体名等）。
        这里以简易方法示例，可用更加复杂或领域适配的NLP工具。
        """
        # 伪示例：用正则等简单手段抽取含“重点”或“重要信息”关键词
        # 实际中往往会使用 jieba / HanLP / 自定义词库进行更精准的抽取
        pattern = r'(重点|重要信息|关键数据)'
        matches = re.findall(pattern, text)
        # 将匹配到的词进行去重
        key_terms = list(set(matches))
        return key_terms

    def check_summary_coverage(self, summary_text: str, key_terms: List[str]) -> Dict[str, float]:
        """
        检查最终摘要是否包含原文大部分关键信息。
        返回结果包含缺失率等指标。
        """
        if not key_terms:
            return {"missing_count": 0, "total_key_count": 0, "missing_rate": 0.0}

        missing_count = 0
        for term in key_terms:
            # 这里以简单的字符串包含为例，可用更高级的相似度判断
            if term not in summary_text:
                missing_count += 1
        
        missing_rate = missing_count / len(key_terms)
        return {
            "missing_count": missing_count,
            "total_key_count": len(key_terms),
            "missing_rate": missing_rate
        }

    def summarize_document_with_check(self, doc_id: str, text: str, missing_threshold: float = 0.2) -> str:
        """
        主流程：分片摘要 + 二次汇总 + 关键信息对比校验。
        若缺失率过高，可再次尝试重生成或人工校对。
        """
        # 1. 分片
        chunks = self.split_document(text)
        
        # 2. 分片摘要
        chunk_summaries = self.summarize_chunks(chunks)

        # 3. 二次汇总
        if len(chunk_summaries) > 1:
            final_summary = self.second_round_summary(chunk_summaries)
        else:
            final_summary = chunk_summaries[0]

        # 4. 关键信息抽取
        key_terms = self.extract_key_info(text)

        # 5. 关键信息覆盖校验
        coverage_info = self.check_summary_coverage(final_summary, key_terms)
        if coverage_info["missing_rate"] > missing_threshold:
            # 缺失太多，说明摘要可能舍弃了重要信息
            # 可以根据实际策略，二次提示模型包含关键信息，或触发人工检查，这里做简单示例
            print(f"Warning: Summaries missing too many key terms. Missing rate = {coverage_info['missing_rate']:.2f}")
            # 通过更明确的提示语调用模型重新生成摘要，或人工介入
            # 下面演示一次简单的重试：
            prompt_hint = "请在摘要中保留以下关键信息:" + ",".join(key_terms)
            # 简易实现：将提示语附加为上下文，再调用模型
            final_summary = self.call_qwen_model(prompt_hint + "\n" + text)

        # 6. 向量化 + 存储ES
        summary_vector = self.vectorize_text(final_summary)
        self.save_to_es(doc_id, final_summary, summary_vector, len(chunks))

        return final_summary
```

---

### 代码说明

1. extract_key_info():  
   - 演示从原文中抽取关键信息的伪实现；  
   - 在实际场景中，可替换成更专业、复杂的 NLP 模块（如词性标注、实体识别、主题短语抽取等）。  

2. check_summary_coverage():  
   - 以简单的“字符串包含”检查方式评估摘要对关键字的覆盖情况；  
   - 若缺失过多，则说明摘要可能丢失了重要内容，可用向量相似度或近义词匹配提升准确性。  

3. summarize_document_with_check():  
   - 整合分片、摘要、二次汇总、校验等操作；  
   - 如果发现缺失率（missing_rate）超出给定阈值 (missing_threshold)，会发出警告并进行重试或进一步干预（例如：给模型更明确的提示；引入人工审阅流程；或者运用其他更严格的摘要策略）。

通过以上逻辑，就可以在自动摘要流程中对模型输出做一定程度的“原文比对”，减小信息损耗的风险。若用在实际生产环境，还可以在“key_terms”提取环节中进一步加强对特定领域关键信息（例如法律法规、财务数字、专有名词等）的识别，以保证总结时这些重要信息不会被忽视。

# Q4：所以你的意思是，输入一些【关键词】，然后通过【用一定策略（字符串匹配、词频/短句匹配、向量相似度等）在模型摘要中进行搜索，判断是否“缺失”】，那具体介绍这些策略如何实现
以下从「字符串匹配」「词频/短句匹配」以及「向量相似度」等三个方向，分别阐述能够检验关键词在摘要中是否被有效保留的不同策略，并通过示例代码说明如何在实际项目中实现这些逻辑，从而减少重要信息的丢失。

---

## 一、字符串匹配策略

最简单直接的方法：将从原文中提取出来的关键词（或知识点）逐一与「模型摘要」进行字符串匹配。若关键词完全以原样出现在摘要中，则视为“被保留”，否则视为“缺失”。

### 1.1 实现原理

1. 从原文中抽取关键词（可基于业务词典、正则规则或实体识别等）。  
2. 遍历摘要文本，若摘要包含该关键字（完全一致的子字符串），则判定这一关键字已被保留，否则算缺失。  
3. 统计缺失率（缺失个数 / 总关键字数量）。若超出阈值，说明摘要丢失过多核心信息，可进行重试或提示。

优点：  
- 简单易实现、速度快。  
缺点：  
- 只适合关键词“原文不变”的匹配场景；若模型摘要有改写（同义替换或缩写），可能导致错判。

### 1.2 参考代码示例

```python:services/string_match_checker.py
def check_string_match(summary_text: str, key_terms: list) -> float:
    """
    利用字符串匹配检查 keywords 是否在 summary 中被保存
    返回缺失率(0~1)
    """
    if not key_terms:
        return 0.0

    missing_count = 0
    for term in key_terms:
        if term not in summary_text:
            missing_count += 1
    
    missing_rate = missing_count / len(key_terms)
    return missing_rate
```

---

## 二、词频/短句匹配策略

在有些场景下，不仅要判断“是否出现”，还可能要关注出现的“次数”或“密度”是否与原文严重偏离。本策略可分为若干变体，例如统计关键词在原文与摘要中的出现频率之比，或进行更灵活的短句匹配。

### 2.1 实现原理

1. 对原文与生成的摘要分别进行分词或短语切割，统计每个关键词出现频次。  
2. 对比原文跟摘要中该关键词出现次数的比率，一旦 ratio 过低，说明该关键词在摘要中被“削减”过度。  
3. 也可将相邻词串起来做“短句匹配”，如果原文中的某些关键短句完全在摘要中消失，也可视作丢失较大。

优点：  
- 能体现“核心词汇”在摘要中的保留程度，而不仅是“出现/没出现”二元判断。  
缺点：  
- 对同义替换或语义变体仍不敏感。  
- 若原文与摘要在长度上相差较大，会需要加一定归一化处理（除以文档总词数、摘要总词数等）。

### 2.2 参考代码示例（简易词频匹配）

```python:services/frequency_checker.py
import re
from collections import Counter
from typing import List

def tokenize(text: str) -> List[str]:
    """
    简单的切词示例，实际可用 jieba 或 HanLP 等更专业的分词。
    """
    # 用正则拆分出中文或英文单词
    tokens = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z0-9]+', text)
    return tokens

def check_keyword_frequency(original_text: str, summary_text: str, key_terms: List[str]) -> float:
    """
    对比 original_text 与 summary_text 中的关键字出现次数比率
    返回“缺失度”或“保留度”的数值
    """
    if not key_terms:
        return 0.0
    
    # 分词
    original_tokens = tokenize(original_text)
    summary_tokens = tokenize(summary_text)

    # 统计词频
    original_counts = Counter(original_tokens)
    summary_counts = Counter(summary_tokens)

    # 计算关键字在原文和摘要中的总出现次数比
    missing_score_sum = 0.0
    for term in key_terms:
        original_freq = original_counts.get(term, 0)
        summary_freq = summary_counts.get(term, 0)
        # 原文中若某关键词频次为0，跳过或按照业务需要处理
        if original_freq == 0:
            continue
        # 比率越小，越可能丢失
        ratio = summary_freq / original_freq
        if ratio < 1.0:
            # 可以累加缺失分，以评估此关键词在摘要中被 “压缩” 了多少
            missing_score_sum += (1.0 - ratio)
    
    # 简化处理：返回平均缺失分
    avg_missing_score = missing_score_sum / len(key_terms)
    return avg_missing_score
```

- 在此实现中：  
  - avg_missing_score 越高，说明摘要对关键词的“保留度”越低；也可以改用其它归一化逻辑。  
  - 如果 avg_missing_score 超过某个阈值（如 0.5），说明该摘要对关键信息的保留不够，可触发重试或人工校验。

---

## 三、向量相似度策略

针对模型摘要可能会做改写（同义替换、语义变形等），用纯字符串或频率匹配难以覆盖所有情况。这时可以将关键词（或短句）与摘要断句后的每一句或每一小段做向量化，并计算余弦相似度。若关键短语在摘要的语义空间里能找到足够高相似度的句子，就判定其已被保留。

### 3.1 实现原理

1. 对从原文抽取出来的每个关键词（或关键短句）做句向量（embedding）计算；  
2. 把摘要分句（或分片），对每段也计算句向量；  
3. 计算关键短语向量与摘要各段向量之间的余弦相似度，如若大于一定阈值（例如 0.6~0.8），则视为“在摘要中得到保留或表达”。  
4. 若对于多个关键短语都无法在摘要中找到合格的相似度，则说明信息损耗较大。

### 3.2 参考代码示例（余弦相似度）

下面的范例假设已经有一个“embedding_service”来提供向量化功能，并使用了常见的“句子余弦相似度”措施。

```python:services/vector_similarity_checker.py
import math
from typing import List

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算余弦相似度
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

class EmbeddingService:
    def embed_sentence(self, text: str) -> List[float]:
        """
        模拟将文本转换成向量表示的接口
        实际可调用BERT、RoBERTa或其他embedding模型进行向量化
        """
        # 这里仅做简单Mock
        return [0.01] * 768

def check_vector_similarity(summary_text: str, key_phrases: List[str],
                            embed_service: EmbeddingService,
                            threshold: float = 0.7) -> float:
    """
    计算关键短语在摘要中的向量相似度是否超过指定阈值。
    返回缺失率(0~1)。缺失率越高，表示越多关键短语“不足够”在摘要中得到保留。
    """
    if not key_phrases:
        return 0.0

    # 1. 分句(或分段)摘要文本
    summary_sentences = summary_text.split('。')  # 简单以句号分割，可更灵活
    
    # 2. 为摘要的每个句子生成向量
    summary_vectors = [embed_service.embed_sentence(sentence.strip()) for sentence in summary_sentences]

    # 3. 对每个关键短语做embedding，再与summary中所有句子计算相似度
    missing_count = 0
    for phrase in key_phrases:
        phrase_vec = embed_service.embed_sentence(phrase)
        # 找到摘要里最高的相似度
        max_sim = 0.0
        for svec in summary_vectors:
            sim = cosine_similarity(phrase_vec, svec)
            if sim > max_sim:
                max_sim = sim
        
        # 若最高相似度仍低于阈值，则视为缺失
        if max_sim < threshold:
            missing_count += 1

    missing_rate = missing_count / len(key_phrases)
    return missing_rate
```

在该示例中：  
- 会将摘要文本按照简单的句号做分割，可以进一步细化或使用 NLP 工具进行更精准的分句。  
- 将关键短语和摘要中的各句分别求向量，然后计算余弦相似度。只要能找到某个句子与关键短语的相似度超过阈值，即可判定“该关键短语已被保留（语义意义相当）”。  
- 最终通过 missing_rate 监控丢失量，如果超过设定的阈值（比如 0.3），就说明至少 30% 的关键短语在摘要中没有得到合理表达。

---

## 四、策略对比与应用建议

1. 字符串匹配：  
   - 简单快速，但对改写、同义词等情况不敏感。适合“必须出现原文关键词”或“专业术语不能变形”的场景。  

2. 词频/短句匹配：  
   - 适合衡量摘要中保留的关键词是否与原文大致“数量对等”，但仍会漏掉改写、翻译等情况。  

3. 向量相似度：  
   - 最通用、灵活度高，能应对改写或同义表达，但需要预先有可靠的嵌入模型或调用第三方服务（性能开销更大）。适合对“语义保真度”有高要求的场景。  

在实际使用中，往往会结合多种策略：  
• 先基于字符串匹配和词频对一些非常核心的“专有名词”做强校验；  
• 再用向量相似度检查更泛化的短语或表达是否基本被涵盖；  
• 最后若模型自由改写或总结较为激进，可在发现缺失严重时给模型新的提示（Prompt），告诉它在摘要中必须保留或体现这些关键信息，以保证最终质量。

---

## 五、小结

• 通过以上三种策略的组合，可以有效衡量模型摘要在保留原文关键信息方面的质量。  
• 当检测到信息缺失率过高时，可触发重试、人工审阅或进一步的二次提示，引导模型优先保留这些重点内容。  
• 不同策略适合不同业务场景，可按需灵活搭配，以在摘要准确度与性能成本间取得平衡。
