# 【论文解读】MEMORAG：通过受记忆启发的知识发现迈向下一代RAG

## 1. 论文基础信息

- **发布机构**：北京人工智能学院、人民大学高岭人工智能学院
- **发布日期**：2024年9月10日
- **代码仓库**：https://github.com/qhjqhj00/MemoRAG
- **简介**：**MemoRAG** 关键是创建了**记忆模块**：
     - 使用一个**轻量级**但具备长范围处理能力的大语言模型（LLM）来构建数据库的全局记忆。
     - 任务提出后，生成**草稿答案**，提供检索线索，以定位数据库中有用的信息。
   
## 2. 核心问题

现有的检索增强生成（RAG）系统在处理信息**需求模糊或非结构化知识的任务**时存在固有限制。这些系统主要依赖于明确查询与结构化知识之间的相关性匹配，导致在复杂任务或需要高层理解的应用场景下表现不佳，仅能有效应对简单的问答任务。

## 3. 论文提出的解决方案🔍
![](https://files.mdnice.com/user/9391/bc77c80e-abe6-46c0-880d-01892d13d2a2.png)

![](https://files.mdnice.com/user/9391/b15bd063-2607-453c-be2d-c90da40101ec.png)

**MemoRAG** 是对标准 **RAG**（**Retrieval-Augmented Generation**）框架的改进，旨在通过引入**内存模块**来弥补标准检索器在理解隐含查询意图时的不足。

### 3.1. 标准RAG框架
标准的RAG框架可以简洁地表示为：

$$
Y = Θ(q, C | θ), \quad C = Γ(q, D | γ)
$$

- **Θ**：生成模型
- **Γ**：检索模型
- **q**：输入查询
- **C**：从数据库**D**中检索的上下文
- **Y**：最终答案

然而，标准RAG在处理隐含信息需求时往往力不从心，因为它主要依赖于词汇或语义匹配，难以全面理解复杂意图。

### 3.2. MemoRAG 的改进
**MemoRAG** 引入了一个**内存模型** **Θmem(·)**，作为输入查询**q**和相关数据库**D**之间的语义桥梁。其流程可表示为：

$$
Y = Θ(q, C | θ), \quad C = Γ(y, D | γ), \quad y = Θmem(q, D | θ_{mem})
$$

**重点公式：**
$$
y = Θmem(q, D | θ_{mem})
$$

- **y**：阶段性答案，提供指导检索相关上下文的线索
- **Θmem(·)**：内存模型，用于建立数据库**D**的全局内存

### 3.3. 内存模块详解
内存模块的设计使其能够将原始输入**X**压缩成较小的**内存token**，同时保留关键信息。具体过程如下：

1. **输入转换**
   $$
   Q = XW_Q, \quad K = XW_K, \quad V = XW_V
   $$
   $$
   Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$
   
2. **引入内存Token**
   $$
   X = \{x_1, \ldots, x_l, x_{m1}, \ldots, x_{mk}, x_{l+1}, \ldots\}, \quad k \ll l
   $$
   $$
   Q_m = XW_{Qm}, \quad K_m = XW_{Km}, \quad V_m = XW_{Vm}
   $$
   $$
   Attention(Q, K, V) = softmax\left(\frac{[Q; Q_m][K; K_m; K_{m\ cache}]^T}{\sqrt{d_k}}\right)[V, V_m, V_{m\ cache}]
   $$

**详细推导：**

1. **查询、键、值的计算：**
   - $Q = XW_Q$
   - $K = XW_K$
   - $V = XW_V$

2. **注意力机制：**
   $$
   Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$

3. **引入内存Token后的注意力机制：**
   - 计算内存Token的查询、键、值：
     $$
     Q_m = XW_{Qm}, \quad K_m = XW_{Km}, \quad V_m = XW_{Vm}
     $$
   - 修改后的注意力计算：
     $$
     Attention(Q, K, V) = softmax\left(\frac{[Q; Q_m][K; K_m; K_{m\ cache}]^T}{\sqrt{d_k}}\right)[V, V_m, V_{m\ cache}]
     $$

### 3.4. 系统实现
**MemoRAG** 系统支持多种**检索方法**（如稀疏检索、密集检索和重排序），默认使用**密集检索**。内存模块目前包括 **memorag-qwen2-7b-inst** 和 **memorag-mistral-7b-inst**，基于不同的语言模型。

- **硬件支持**：
  - **NVIDIA T4 16GiB**：处理上下文长度68K tokens
  - **NVIDIA A100 80GiB**：处理上下文长度高达100万 tokens

### 3.5. 应用场景

#### 1. 处理隐含查询意图
标准RAG在处理隐含意图时可能表现不佳，而**MemoRAG**通过**全局内存**生成**答案线索**，有效提升检索相关内容的能力。

**示例 - 表1：处理隐含查询**
```
查询：How does the book convey the theme of love?
数据库：Harry Potter series
线索 #1：Lily Potter’s sacrifice
线索 #2：The Weasley Family
线索 #3：Romantic relationship with Ginny Weasley
答案：...
```

#### 2. 多跳查询
**MemoRAG** 能够生成**阶段性答案**，拆分复杂查询，整合跨步骤的信息，提升多跳推理的效果。

**示例 - 表2：处理多跳查询**
```
查询：Which year had the peak revenue in the past three years?
数据库：Past ten years’ financial reports...
```

#### 3. 信息聚合
在需要汇总大量非结构化数据的任务中，**MemoRAG** 利用内存模块提取关键信息点，生成连贯的总结。

**示例 - 表3：信息聚合**
```
任务：Summarize the government report
数据库：A government report on city construction
```

#### 4. 个性化助手
通过分析用户对话历史，**MemoRAG** 生成高度个性化的推荐，提升用户体验。

**示例 - 表4：个性化推荐**
```
查询：Can you recommend a song for me?
数据库：Dialogue history of a user
```

#### 5. 终身对话搜索
**MemoRAG** 通过维护全局对话历史内存，准确理解并响应上下文相关的后续查询。

**示例 - 表5：对话搜索**
```
查询：Does it have any weaknesses compared the paper we discussed last Monday?
数据库：Conversational search history...
```

## 4. 实验分析 📊

### 4.1. 数据集

使用13个现有的基准数据集，包括：
<table border="1" cellspacing="0" cellpadding="5">  
    <tr>  
        <th>任务</th>  
        <th>数据集</th>  
        <th>发布来源</th>  
    </tr>  
    <tr>  
        <td rowspan="3">单文档QA</td>  
        <td>NarrativeQA</td>  
        <td>Kociský et al., 2017</td>  
    </tr>  
    <tr>  
        <td>Qasper</td>  
        <td>Dasigi et al., 2021</td>  
    </tr>  
    <tr>  
        <td>Multi-FieldQA</td>  
        <td>Bai et al., 2023</td>  
    </tr>  
    <tr>  
        <td rowspan="3">多文档QA</td>  
        <td>HotpotQA</td>  
        <td>Yang et al., 2018</td>  
    </tr>  
    <tr>  
        <td>2WikiMQA</td>  
        <td>Ho et al., 2020</td>  
    </tr>  
    <tr>  
        <td>MuSiQue</td>  
        <td>Trivedi et al., 2022</td>  
    </tr>  
    <tr>  
        <td rowspan="3">摘要任务</td>  
        <td>GovReport</td>  
        <td>Huang et al., 2021</td>  
    </tr>  
    <tr>  
        <td>MultiNews</td>  
        <td>Fabbri et al., 2019</td>  
    </tr>  
    <tr>  
        <td>En.SUM</td>  
        <td>Zhang et al., 2024</td>  
    </tr>  
    <tr>  
        <td>长篇小说QA</td>  
        <td>En.QA</td>  
        <td>Zhang et al., 2024</td>  
    </tr>  
</table>

**但在实际场景中，不是所有用户查询都有明确的信息需求。大多数查询需要对全文进行全面理解，并整合多个信息片段以获得最终答案。**

为了评估MemoRAG和标准RAG系统在广泛应用中的表现，构建了**ULTRADOMAIN基准**。该基准包括涉及长上下文和高级查询的多个专业领域的任务。

##### **ULTRADOMAIN** 基准的构建
首先，利用代表知识领域的数据集中的上下文，重点关注两个专业数据集：
- **Fin数据集**：源自财务报告，测试MemoRAG处理和解释复杂财务数据的能力，确保系统能够处理财务语言和报告的复杂性。
- **Legal数据集**：源自法律合同，挑战MemoRAG理解并导航法律文件中的复杂和微妙语言，其中精确性至关重要。

除了这些专业数据集，还收集了涵盖**18个不同领域的428本大学教科书，包括自然科学、人文学科和社会科学**。这些教科书用于测试MemoRAG在广泛话题上的多样性和适应性，这些话题可能与专业数据集不直接相关。通过在这些多样化的上下文中评估MemoRAG，更深入地了解了其在特定领域（如金融和法律）之外的潜在应用。

最后，构建了一个包含上述数据集混合上下文的数据集，即Mix。这个混合数据集旨在评估MemoRAG如何在不同类型上下文中泛化其理解能力。

![](https://files.mdnice.com/user/9391/2f076944-56fb-4df3-b0ac-39d46218f80a.png)
上表提供了专业数据集的统计详情，

![](https://files.mdnice.com/user/9391/399f7c57-b3dd-403c-9eec-cfd8557870b1.png)
上表提供了教科书数据集的统计详情。这些数据集共同构成了一个全面的基准，严格测试MemoRAG在处理特定领域挑战和更广泛的跨学科任务方面的有效性。

### 4.2. 基线方法

<table border="1" cellspacing="0" cellpadding="5">  
    <tr>  
        <th>基线方法</th>  
        <th>发布来源</th>  
        <th>简介</th>  
    </tr>  
    <tr>  
        <td>Full</td>  
        <td>—</td>  
        <td>直接将完整上下文输入LLMs，以适应LLMs的最大长度。</td>  
    </tr>  
    <tr>  
        <td>BGE-M3</td>  
        <td>Chen et al., 2023</td>  
        <td>一个通用检索器，用它执行标准RAG。</td>  
    </tr>  
    <tr>  
        <td>Stella-en-1.5B-v5</td>  
        <td>—</td>  
        <td>该模型在撰写本文时在MTEB排行榜上排名第3，用它执行标准RAG。</td>  
    </tr>  
    <tr>  
        <td>RQ-RAG</td>  
        <td>Chan et al., 2024</td>  
        <td>RQ-RAG提示LLMs将输入查询分解成几个更好的查询，以便于进行明确的重写、分解和消歧。支持段落由输入查询和优化查询检索。</td>  
    </tr>  
    <tr>  
        <td>HyDE</td>  
        <td>Gao et al., 2022</td>  
        <td>直接提示LLMs通过仅提供查询生成假文档，然后使用假文档检索段落，最终根据检索的段落生成答案。</td>  
    </tr>  
</table>

为了更全面的比较，使用了三种流行的LLMs作为生成器：
- **Llama3-8B-Instruct-8K**
- **Mistral-7B-Instruct-v0.2-32K** (Jiang et al., 2023)
- **Phi-3-mini-128K** (Abdin et al., 2024)

### 4.3. ULTRADOMAIN实验

大多数ULTRADOMAIN查询涉及模糊的信息需求或非结构化知识检索挑战。ULTRADOMAIN包括两种类型的数据集：
1. **第一类**：包括三个上下文大小在100K tokens以下的数据集，这些数据集与的训练数据集分布相同。称这些数据集为**in-domain**数据集。
2. **第二类**：包括18个数据集，这些数据集源自428本英文大学教科书，上下文长度可达一百万tokens。由于第二类数据集的数据分布与训练数据不同，称其为**out-of-domain**数据集。

实验结果总结在下表中，从中可以得出以下结论：

![](https://files.mdnice.com/user/9391/7d1ad6a9-c299-423c-bf43-7fa51cd59002.png)


1. **MemoRAG在所有数据集上均优于所有基线模型**，展示了其强大的领域泛化能力。
2. **直接将完整上下文输入LLMs通常比其他RAG方法（BGE-M3、Stella-v5和HyDE）表现更好**。这一发现表明，标准RAG系统在处理长上下文和高级问题时存在困难。
3. **相比之下，MemoRAG在处理完整上下文时始终超越其他方法**，展示了其有效弥合处理超级长上下文和解决复杂任务之间差距的能力。
4. **MemoRAG在三个in-domain数据集上表现出色**，表明其潜力可以通过更多样化的训练数据进一步增强。

### 4.4. 所有基准对比实验


![](https://files.mdnice.com/user/9391/f8fc7410-2d06-49bc-9eb5-bc3477f1336a.png)
上表显示了三个基准上的实验结果，从中可以得出以下结论：
1. **MemoRAG在所有数据集上普遍优于所有基线模型**，只有一个异常值除外。
2. **对于开放域QA任务**，MemoRAG在所有数据集上均优于所有基线模型，除了使用Llama3生成器的en.qa数据集。这验证了在标准RAG的舒适区（大多数查询有明确的信息需求）中，MemoRAG能够更好地在原始上下文中定位预期的证据，这得益于记忆生成的线索。
3. **大多数先前的RAG方法在不涉及查询的任务中表现不佳**，例如摘要任务（例如MultiNews、GovReport和en.sum）。的MemoRAG使RAG系统能够从输入上下文中生成关键点，并检索更多细节以形成全面的摘要。
4. **对于特定领域的任务**（例如金融和法律），MemoRAG表现出显著的改进，表明其在处理涉及长上下文的复杂任务时具有优势。

---

# 【项目解读】MemoRAG
* https://github.com/qhjqhj00/MemoRAG
## 1. 核心类图
![](https://files.mdnice.com/user/9391/a7d0975c-d9b1-481f-a4b4-f2d0c7585097.png)

## 2. 代表性问题

### 1. MemoRAG 如何处理超长上下文?

根据源码,MemoRAG主要通过以下机制处理超长上下文:

1. 分块处理:
    - 使用TextSplitter将长文本分成固定大小的chunks
    - 对每个chunk生成gist(摘要)
    - 将所有gist拼接形成压缩后的上下文

```python
    # 172:224:thirdparty/MemoRAG/memorag/memorag_lite.py
    def memorize(
        self, 
        context: str, 
        save_dir: str = None, 
        print_stats: bool = True, 
        batch_size: int = 1,
        gist_chunk_size: int = 4096):
    
        self.reset()

        # Detect language
        text_sample = context[:1024]
        self.language = detect(text_sample)
        if print_stats:
            print(f"Detected language: {self.language}")

        batch_size = self.adapt_batch_size()

        # Encode context
        encoding = tiktoken.get_encoding("cl100k_base")
        encoded_context = encoding.encode(context)
        if print_stats:
            print(f"Context length: {len(encoded_context)} tokens")

        # Set appropriate prompts based on detected language
        self.prompts = zh_prompts if self.language == "zh-cn" else en_prompts

        # Split context into gists
        text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", gist_chunk_size)
        gist_chunks = text_splitter.chunks(context)
        gist_chunks = [self.prompts["gist"].format(context=chunk) for chunk in gist_chunks]

        # Generate gists
        if print_stats:
            print(f"Forming memory of the context...")

        self.gists = []
        for i in range(0, len(gist_chunks), batch_size):
            if print_stats and i > 1:
                progress = round(i / len(gist_chunks) * 100, 2)
                print(f"Progress: {progress}% of the context memorized...")

            gists_batch = self.gen_model.generate(
                gist_chunks[i:i+batch_size], 
                batch_size=batch_size, 
                max_new_tokens=512, 
                repetition_penalty=1.2)
            torch.cuda.empty_cache()
            
            self.gists.extend(gists_batch)
        # Join generated gists and clear cache
        gists_concatenated = "\n".join(self.gists)
```


2. 双重索引机制:
    - 使用DenseRetriever建立稠密检索索引
    - 检索chunk大小可配置,中文默认2048,英文默认512

```python
        # 346:349:thirdparty/MemoRAG/memorag/memorag.py
        self.retriever = DenseRetriever(
            ret_model_name_or_path, hits=ret_hit, cache_dir=cache_dir, load_in_4bit=load_in_4bit)

        self.text_splitter = TextSplitter.from_tiktoken_model(
```



### 2. 如何整合关键点生成摘要?
1. 调用Memory模块的summarize()获取关键点
2. 过滤关键点,保留有效信息
3. 基于关键点检索相关文本片段
4. 将检索结果整合生成最终摘要
MemoRAG通过以下步骤整合信息:

```python
    # 437:444:thirdparty/MemoRAG/memorag/memorag.py
    def _handle_summarization(self, prompt_template: str, max_new_tokens: int):
        key_points = self.mem_model.summarize()
        retrieval_query = [query for query in key_points.split("\n") if len(query.split()) > 3]

        retrieval_results = self._retrieve(retrieval_query)
        knowledge = "\n\n".join(retrieval_results)

        return self._generate_response("sum_gen", None, knowledge, prompt_template, max_new_tokens)
```


### 3. 如何平衡计算资源与性能?

源码中采用了多种优化策略:

1. 显存自适应:
    - 根据GPU显存动态调整batch size
    - 针对不同语言设置不同的阈值
```python
    # 153:169:thirdparty/MemoRAG/memorag/memorag_lite.py
    def adapt_batch_size(self):
        free_memory = get_first_gpu_memory()

        if free_memory < 23000:
            print(f"The minimum recommended GPU memory for MemoRAG is 24GiB, but only {round(free_memory / 1024, 1)} GiB is available.")

        if self.adapt_bs:
            memory_thresholds = {
                "en": [(70000, 16), (60000, 10), (38000, 8), (20000, 4), (14000, 2)], 
                "zh-cn": [(70000, 16), (60000, 10), (38000, 8), (20000, 4), (14000, 2)]  
            }
            thresholds = memory_thresholds.get(self.language, memory_thresholds["en"])

            for threshold, bs in thresholds:
                if free_memory > threshold:
                    batch_size = bs
                    break
```


2. 量化优化:
    - 支持4bit量化加载模型
    - 可配置是否启用flash attention
```python
        # 62:66:thirdparty/MemoRAG/memorag/memorag.py
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit
                )
            self.model_kwargs["quantization_config"] = quant_config
```


3. 缓存管理:
    - 及时清理GPU缓存
    - 支持保存/加载记忆到磁盘
```python
        # 289:291:thirdparty/MemoRAG/memorag/memorag_lite.py
        memory_size_gb = os.path.getsize(memory_path) / (1024 ** 3)
        print(f"Memory file size: {memory_size_gb:.2f} GB")
        print(f"Number of chunks in retrieval corpus: {len(self.retrieval_corpus)}")
```

总的来说,MemoRAG通过分块处理、双重索引、动态batch size等机制,在有限资源下实现了高效的长文本处理。同时提供了丰富的配置选项,可以根据实际场景调整资源使用。