# fastRAG：构建和探索高效的检索增强生成模型和应用


![](https://files.mdnice.com/user/9391/1ce39fbd-33a1-434a-9f38-936f8c78c905.png)

## 0. 简介

### 主要特点
- **优化RAG**：使用SOTA高效组件构建RAG管道，以提高计算效率。
- **针对英特尔硬件进行了优化**：利用针对PyTorch（IPEX）的英特尔扩展、🤗最佳英特尔和🤗最佳哈瓦那在英特尔®至强®处理器和英特尔®高迪®人工智能加速器上尽可能最佳地运行。
- **可定制**：Fast RAG是使用Haystack和HuggingFace构建的。所有Fast RAG的组件都100%兼容Haystack。

![](https://files.mdnice.com/user/9391/c8a38f80-5d6b-49e7-91e9-de674a180252.png)

## 1. 项目的架构设计

### 1.1 系统整体架构

fastRAG是一个基于Haystack框架的RAG系统实现，主要包含以下核心模块：

![](https://files.mdnice.com/user/9391/5dec8b9e-6d8b-4c1f-bccd-d5192119188c.png)

```plantuml
@startuml FastRAG System Architecture

package "FastRAG Core" {
  [Agents] as agents
  [Generators] as generators
  [Rankers] as rankers 
  [Retrievers] as retrievers
  [Stores] as stores
}

package "UI Layer" {
  [Chainlit UI] as ui
}

package "External Services" {
  [LLM Services] as llm
  [Vector Stores] as vector
}

agents --> generators
agents --> retrievers
agents --> rankers
retrievers --> stores
generators --> llm
stores --> vector

ui --> agents

@enduml
```

### 1.2 核心包的类图设计

#### 1.2.1 Agents包类图


![](https://files.mdnice.com/user/9391/15cbdb06-04dd-409e-9736-00c31b9a2e83.png)
类图主要展示了:

1. 核心组件:
    - Agent: 主要的agent类，负责协调工具和内存
    - ConversationMemory: 对话记忆管理
    - ToolsManager: 工具管理器
    - Tool: 基础工具类

2. 处理器组件:
    - AgentTokenStreamingHandler: Token流处理
    - HFTokenStreamingHandler: HuggingFace Token处理

3. 回调组件:
    - HaystackAgentCallbackHandler: 代理回调处理

4. 关键关系:
    - 组合关系: Agent与Memory/ToolsManager
    - 继承关系: Tool继承自HaystackPipelineContainer
    - 依赖关系: Handler之间的依赖

这个类图完整展示了agents包的核心架构和组件关系。

```plantuml
@startuml Agents Package

' 基础类和接口
abstract class StorageNameSpace {
  + namespace: str
  + global_config: dict
  + {abstract} index_done_callback()
  + {abstract} query_done_callback()
}

class Agent {
  + memory: ConversationMemory
  + tools_manager: ToolsManager
  + tokenizer: AutoTokenizer
  + stopping_criteria: StoppingCriteriaList
  + __call__(query: str, params: dict)
  + run(query: str, params: dict)
  + get_next_action()
  + execute_action()
}

class ConversationMemory {
  - messages: List[dict]
  + add_message(role: str, content: str)
  + get_messages(): List[dict]
  + clear()
}

class ToolsManager {
  - tools: Dict[str, Tool]
  + register_tool(tool: Tool)
  + get_tool(name: str): Tool
  + get_tools_description(): str
}

class Tool {
  + name: str
  + description: str
  + pipeline: Pipeline
  + __call__(params: dict)
  + load_pipeline(pipeline_or_yaml_file)
}

' 处理器类
class AgentTokenStreamingHandler {
  + callback_manager: Events
  + on_new_token(token: str)
}

class HFTokenStreamingHandler {
  + tokenizer: AutoTokenizer
  + stream_handler: AgentTokenStreamingHandler
  + put(value: Any)
  + end()
}

' 工具实现类
class HaystackPipelineContainer {
  + load_pipeline(pipeline_or_yaml_file): Pipeline
}

' 回调处理类
class HaystackAgentCallbackHandler {
  - agent_name: str
  - stack: Stack
  - last_step: Step
  - stream_final_answer: bool
  + on_agent_start()
  + on_agent_step()
  + on_agent_finish()
  + on_new_token()
}

' 关系定义
Agent --> ConversationMemory
Agent --> ToolsManager
ToolsManager o--> Tool
Tool --|> HaystackPipelineContainer

AgentTokenStreamingHandler --> "Events" Events
HFTokenStreamingHandler --> AgentTokenStreamingHandler

HaystackAgentCallbackHandler --> Agent

@enduml
```
#### 1.2.2. embedders包类图

![](https://files.mdnice.com/user/9391/dbddb3b0-9c7f-4020-8348-e793c76dfc57.png)
```plantuml
@startuml
' 基础接口/类
package "haystack.components.embedders" {
  abstract class SentenceTransformersDocumentEmbedder {
    + model_name_or_path: str
    + model_kwargs: dict
    + token: Secret
    + prefix: str
    + suffix: str
    + device: ComponentDevice
    + embed_documents(documents: List[Document]): List[List[float]]
  }

  abstract class SentenceTransformersTextEmbedder {
    + model_name_or_path: str
    + model_kwargs: dict
    + token: Secret
    + device: ComponentDevice
    + embed_texts(texts: List[str]): List[List[float]]
  }
}

package "fastrag.embedders" {
  class IPEXSentenceTransformersDocumentEmbedder {
    + model_name_or_path: str
    + model_kwargs: dict
    + token: Secret
    + device: ComponentDevice
    + __init__()
    + embed_documents()
    - _initialize_model()
  }

  class IPEXSentenceTransformersTextEmbedder {
    + model_name_or_path: str
    + model_kwargs: dict
    + token: Secret
    + device: ComponentDevice
    + __init__()
    + embed_texts()
    - _initialize_model()
  }

  class _SentenceTransformersEmbeddingBackend {
    + model: IPEXModel
    + tokenizer: AutoTokenizer
    + __init__()
    + embed()
    - _initialize_model()
  }
}

' 继承关系
SentenceTransformersDocumentEmbedder <|-- IPEXSentenceTransformersDocumentEmbedder
SentenceTransformersTextEmbedder <|-- IPEXSentenceTransformersTextEmbedder

' 依赖关系
IPEXSentenceTransformersDocumentEmbedder ..> _SentenceTransformersEmbeddingBackend
IPEXSentenceTransformersTextEmbedder ..> _SentenceTransformersEmbeddingBackend

note right of IPEXSentenceTransformersDocumentEmbedder
  使用Intel IPEX优化的文档嵌入器
end note

note right of IPEXSentenceTransformersTextEmbedder
  使用Intel IPEX优化的文本嵌入器
end note
@enduml
```
#### 1.2.2 Generators包类图


![](https://files.mdnice.com/user/9391/d59ce6e0-d0ff-4531-90b3-4bd45334f74d.png)

类图主要展示了:
1. 所有生成器继承自抽象基类 BaseGenerator
2. 每个具体生成器针对不同的硬件或框架进行了优化:
   - FiD: Fusion-in-Decoder生成
   - Gaudi: Habana Gaudi加速器支持
   - IPEX: Intel PyTorch扩展优化
   - Llava: 多模态(视觉-语言)生成
   - OpenVINO: Intel推理引擎优化
   - ORT: ONNX Runtime优化
   - Replug: 可插拔式生成器
3. 统一的生成接口设计,但支持不同的硬件加速和优化方案

```plantuml
@startuml Generators

' 基础生成器
abstract class BaseGenerator {
  + model: PreTrainedModel
  + tokenizer: AutoTokenizer
  + device: str
  + {abstract} generate(prompt: str): str
  + warm_up()
}

' 具体生成器实现
class FiDGenerator {
  + model_name: str
  + generation_kwargs: dict
  + __init__(model_name: str, **kwargs)
  + generate(prompt: str): str
}

class GaudiGenerator {
  + model_name: str
  + habana_device: str
  + generate(prompt: str): str
}

class IPEXGenerator {
  + model_name: str
  + ipex_config: dict
  + generate(prompt: str): str
}

class LlavaHFGenerator {
  + vision_tower: PreTrainedModel
  + projection_layer: nn.Linear
  + process_images(images: List)
  + generate(prompt: str, images: List): str
}

class OpenVINOGenerator {
  + model_path: str
  + config: dict
  + generate(prompt: str): str
}

class ORTGenerator {
  + session: InferenceSession
  + generate(prompt: str): str
}

class ReplugGenerator {
  + model_name: str
  + replug_config: dict
  + generate(prompt: str): str
}

' 继承关系
BaseGenerator <|-- FiDGenerator
BaseGenerator <|-- GaudiGenerator 
BaseGenerator <|-- IPEXGenerator
BaseGenerator <|-- LlavaHFGenerator
BaseGenerator <|-- OpenVINOGenerator
BaseGenerator <|-- ORTGenerator
BaseGenerator <|-- ReplugGenerator

@enduml
```
#### 1.2.3. prompt_builders包

![](https://files.mdnice.com/user/9391/b3f15ab9-3523-4359-9f2c-b6268992eb87.png)

类图主要展示了prompt_builders包中的核心组件设计，包括:
1. 基础的提示词构建器接口(BasePromptBuilder)
2. 三种主要的提示词构建器实现(RAG/对话式/多模态)
3. 提示词压缩器(PromptCompressor)和优化器(PromptOptimizer)的辅助功能
4. 各个组件之间的继承和依赖关系，体现了提示词构建的模块化设计
```plantuml
@startuml PromptBuilders

interface BasePromptBuilder {
  + build_prompt(query: str, context: str): str
  + build_system_prompt(): str
}

class RAGPromptBuilder {
  - template: str
  - system_template: str
  + build_prompt(query: str, context: str): str
  + build_system_prompt(): str
}

class ConversationalPromptBuilder {
  - history: List[Dict]
  - template: str
  - system_template: str
  + build_prompt(query: str, context: str): str
  + build_system_prompt(): str
  + add_to_history(role: str, content: str)
  + clear_history()
}

class MultiModalPromptBuilder {
  - template: str
  - system_template: str
  - image_token: str
  + build_prompt(query: str, context: str, images: List): str
  + build_system_prompt(): str
  + process_images(images: List): str
}

class PromptCompressor {
  - max_tokens: int
  - compression_type: str
  + compress(prompt: str): str
  + chunk_prompt(prompt: str): List[str]
}

class PromptOptimizer {
  - optimization_strategy: str
  + optimize(prompt: str): str
  + rewrite_prompt(prompt: str): str
}

BasePromptBuilder <|-- RAGPromptBuilder
BasePromptBuilder <|-- ConversationalPromptBuilder
BasePromptBuilder <|-- MultiModalPromptBuilder

RAGPromptBuilder --> PromptCompressor
RAGPromptBuilder --> PromptOptimizer
ConversationalPromptBuilder --> PromptCompressor
MultiModalPromptBuilder --> PromptCompressor

@enduml
```
#### 1.2.4. prompt_compressors包

![](https://files.mdnice.com/user/9391/37a53dfd-882f-4c72-9225-2af418c0a64b.png)
类图主要展示了prompt_compressors包中的提示词压缩器类的层次结构。包含一个基础接口BasePromptCompressor定义了**压缩提示词的基本方法**，以及两个具体实现类**LLMLinguaPromptCompressor**和**OVLLMLinguaPromptCompressor**，它们分别实现了基于**LLMLingua**和**OpenVINO**优化的LLMLingua的提示词压缩功能。这种设计使得系统可以灵活切换不同的提示词压缩策略。

```plantuml
@startuml Prompt Compressors

interface BasePromptCompressor {
  + compress(prompt: str): str
  + batch_compress(prompts: List[str]): List[str]
}

class LLMLinguaPromptCompressor {
  - model: str
  - device: str
  - ratio: float
  - min_length: int
  + __init__(model: str, device: str, ratio: float)
  + compress(prompt: str): str
  + batch_compress(prompts: List[str]): List[str]
}

class OVLLMLinguaPromptCompressor {
  - model: str 
  - device: str
  - ratio: float
  - min_length: int
  - model_path: str
  + __init__(model: str, device: str, ratio: float)
  + compress(prompt: str): str
  + batch_compress(prompts: List[str]): List[str]
}

BasePromptCompressor <|-- LLMLinguaPromptCompressor
BasePromptCompressor <|-- OVLLMLinguaPromptCompressor

@enduml
```
#### 1.2.5. rankers包

![](https://files.mdnice.com/user/9391/70be7b5f-f135-45c8-ab43-e8dbbaacdad1.png)
类图主要展示了:
1. Rankers包中的三个主要重排序器实现:BiEncoder、IPEX优化版BiEncoder和ColBERT
2. 它们共同实现的BaseRanker接口,提供rank()和score()方法
3. 各个Ranker与其依赖的外部组件(如Embedder、Tokenizer等)之间的关系
4. 每个具体Ranker的主要属性和方法
```
@startuml Rankers

' 基础接口
interface BaseRanker {
  + rank(documents: List[Document]): List[Document]
  + score(documents: List[Document]): List[float]
}

' 具体实现类
class BiEncoderSimilarityRanker {
  - model: SentenceTransformer
  - device: str
  - batch_size: int
  + __init__(model_name: str, device: str)
  + rank(documents: List[Document]): List[Document]
  + score(documents: List[Document]): List[float]
  - _compute_similarity(query_embedding: tensor, doc_embeddings: tensor): tensor
}

class IPEXBiEncoderSimilarityRanker {
  - model: SentenceTransformer
  - device: str
  - batch_size: int
  + __init__(model_name: str)
  + rank(documents: List[Document]): List[Document]
  + score(documents: List[Document]): List[float]
  - _optimize_for_ipex()
}

class ColBERTRanker {
  - model: ColBERT
  - tokenizer: AutoTokenizer
  - max_length: int
  + __init__(model_name: str, device: str)
  + rank(documents: List[Document]): List[Document]
  + score(documents: List[Document]): List[float]
  - _maxsim(query_tokens: tensor, doc_tokens: tensor): float
}

' 继承关系
BaseRanker <|.. BiEncoderSimilarityRanker
BaseRanker <|.. IPEXBiEncoderSimilarityRanker
BaseRanker <|.. ColBERTRanker

' 关联
BiEncoderSimilarityRanker --> "1" SentenceTransformersDocumentEmbedder: uses
IPEXBiEncoderSimilarityRanker --> "1" SentenceTransformersDocumentEmbedder: uses
ColBERTRanker --> "1" AutoTokenizer: uses

@enduml
```
#### 1.2.6. retrievers包

![](https://files.mdnice.com/user/9391/1b72b0e5-b9e5-457e-a907-9bd26240ccc0.png)

类图主要展示了:
1. 检索器的基础接口BaseRetriever定义了retrieve和add_documents两个核心方法
2. 四种主要的检索器实现:
   - ColBERTRetriever: 基于ColBERT模型的检索
   - DenseRetriever: 基于稠密向量的检索
   - BM25Retriever: 基于BM25算法的稀疏检索
   - HybridRetriever: 混合检索策略,组合了稠密和稀疏检索
3. HybridRetriever通过组合方式复用了DenseRetriever和BM25Retriever的功能
4. 每个检索器都实现了文档的添加和检索两个核心功能
```plantuml
@startuml Retrievers

' 基础接口
interface BaseRetriever {
  + retrieve(query: str, top_k: int): List[Document]
  + add_documents(documents: List[Document])
}

' ColBERT检索器
class ColBERTRetriever {
  - model: PreTrainedModel
  - tokenizer: AutoTokenizer
  - index: ColBERTIndex
  + __init__(model_name: str, device: str)
  + retrieve(query: str, top_k: int): List[Document]
  + add_documents(documents: List[Document])
  - _encode_query(query: str): Tensor
  - _build_index(documents: List[Document])
}

' 向量检索器
class DenseRetriever {
  - embedding_model: SentenceTransformer
  - index: faiss.Index
  + __init__(model_name: str)
  + retrieve(query: str, top_k: int): List[Document]
  + add_documents(documents: List[Document])
  - _encode_documents(documents: List[Document]): ndarray
}

' 混合检索器
class HybridRetriever {
  - dense_retriever: DenseRetriever
  - sparse_retriever: BM25Retriever
  - alpha: float
  + __init__(dense_retriever: DenseRetriever, sparse_retriever: BM25Retriever)
  + retrieve(query: str, top_k: int): List[Document]
  + add_documents(documents: List[Document])
  - _merge_results(dense_results: List, sparse_results: List): List
}

' BM25检索器
class BM25Retriever {
  - index: BM25Okapi
  - tokenizer: Callable
  + __init__(tokenizer: Callable)
  + retrieve(query: str, top_k: int): List[Document]
  + add_documents(documents: List[Document])
  - _tokenize(text: str): List[str]
}

' 继承关系
BaseRetriever <|.. ColBERTRetriever
BaseRetriever <|.. DenseRetriever
BaseRetriever <|.. HybridRetriever
BaseRetriever <|.. BM25Retriever

' 组合关系
HybridRetriever o--> DenseRetriever
HybridRetriever o--> BM25Retriever

@enduml
```
#### 1.2.7. stores包

![](https://files.mdnice.com/user/9391/04ef4fc3-1296-4446-8751-0604eec03cfa.png)
类图主要展示了:
1. 存储模块的三个核心抽象基类:BaseVectorStorage(向量存储)、BaseKVStorage(键值存储)和BaseGraphStorage(图存储)
2. 各个具体实现类如PLAIDDocumentStore、JsonKVStorage、NanoVectorDBStorage和NetworkXStorage的属性和方法
3. 存储类之间的继承关系,所有存储类都继承自StorageNameSpace基类
4. 每个存储类的主要功能接口和实现方法
```plantuml
@startuml Stores

' 基础存储接口
abstract class StorageNameSpace {
  + namespace: str
  + global_config: dict
  + {abstract} index_done_callback()
  + {abstract} query_done_callback()
}

' 向量存储
abstract class BaseVectorStorage {
  + embedding_func: EmbeddingFunc
  + meta_fields: set
  + {abstract} query(query: str, top_k: int): list[dict]
  + {abstract} upsert(data: dict[str, dict])
}

' 键值存储
abstract class BaseKVStorage<T> {
  + {abstract} all_keys(): list[str]
  + {abstract} get_by_id(id: str): T
  + {abstract} get_by_ids(ids: list[str], fields: set[str]): list[T]
  + {abstract} filter_keys(data: list[str]): set[str]
  + {abstract} upsert(data: dict[str, T])
  + {abstract} drop()
}

' 图存储
abstract class BaseGraphStorage {
  + {abstract} has_node(node_id: str): bool
  + {abstract} has_edge(source_node_id: str, target_node_id: str): bool
  + {abstract} node_degree(node_id: str): int
  + {abstract} edge_degree(src_id: str, tgt_id: str): int
  + {abstract} get_node(node_id: str): dict
  + {abstract} get_edge(source_node_id: str, target_node_id: str): dict
  + {abstract} get_node_edges(source_node_id: str): list[tuple]
  + {abstract} upsert_node(node_id: str, node_data: dict)
  + {abstract} upsert_edge(source_node_id: str, target_node_id: str, edge_data: dict)
  + {abstract} embed_nodes(algorithm: str): tuple[np.ndarray, list[str]]
}

' 具体实现类
class PLAIDDocumentStore {
  + __init__(config: dict)
  + add_documents(documents: list)
  + get_documents(ids: list): list
  + search(query: str, top_k: int): list
}

class JsonKVStorage {
  - _file_name: str
  - _data: dict
  + __post_init__()
  + all_keys(): list[str]
  + get_by_id(id): dict
  + get_by_ids(ids, fields): list[dict]
  + filter_keys(data): set[str]
  + upsert(data: dict)
  + drop()
}

class NanoVectorDBStorage {
  - _client: NanoVectorDB
  + query(query: str, top_k: int): list[dict]
  + upsert(data: dict)
  + index_done_callback()
}

class NetworkXStorage {
  - _graph: nx.Graph
  - _graphml_xml_file: str
  + __post_init__()
  + has_node(node_id: str): bool
  + has_edge(source_node_id: str, target_node_id: str): bool
  + get_node(node_id: str): dict
  + get_edge(source_node_id: str, target_node_id: str): dict
  + upsert_node(node_id: str, node_data: dict)
  + upsert_edge(source_node_id: str, target_node_id: str, edge_data: dict)
}

' 继承关系
StorageNameSpace <|-- BaseVectorStorage
StorageNameSpace <|-- BaseKVStorage
StorageNameSpace <|-- BaseGraphStorage

BaseKVStorage <|-- JsonKVStorage
BaseVectorStorage <|-- NanoVectorDBStorage
BaseVectorStorage <|-- PLAIDDocumentStore
BaseGraphStorage <|-- NetworkXStorage

@enduml
```
---

### 1.3 核心功能流程图

![](https://files.mdnice.com/user/9391/35503de4-48f8-4551-83c8-de2d80be37d5.png)

```plantuml
@startuml Core Flow
start

:用户输入查询;

partition "Agent处理" {
  :解析查询意图;
  :选择合适工具;
  
  if (需要检索?) then (yes)
    :调用Retriever检索相关文档;
    :使用Ranker重排序结果;
  endif
  
  :调用Generator生成回答;
}

:返回结果给用户;

stop
@enduml
```

# 2. 设计模式分析

- 工厂模式：在代码中发现工具创建使用了工厂模式
- 策略模式：在生成器实现中使用了策略模式，允许在运行时切换不同的生成策略
- 观察者模式：在UI回调中使用了观察者模式处理事件

## 3. 项目亮点
1. 基于Haystack框架构建,充分利用了其生态系统
2. 模块化设计清晰,各个组件职责单一
3. 支持多模态输入(文本+图像)
4. 提供了灵活的工具系统扩展机制