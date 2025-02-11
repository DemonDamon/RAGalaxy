"""
示例：使用ModernBERT模型对长文本进行多标签分类。
说明：
1. 如果您是从HuggingFace获取ModernBERT，请将 'modernbert-base-chinese' 
   替换为具体的ModernBERT模型名称或本地路径。
2. 该示例使用滑窗分块处理长文本。也可根据需求自行修改。
3. 数据集部分仅示例说明，需自行准备真实数据。
"""

import torch
import torch._dynamo
# 禁用动态编译以避免警告
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True  # 完全禁用动态编译

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------------------
# 1. 数据集与分块工具
# ------------------------------
class LongTextDataset(Dataset):
    """
    由若干长文本构成的数据集，每个文本可能对应多个标签。
    本示例中仅示范输入数据格式与分块逻辑，标签数量与文本分块可自行配置。
    """
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length=512, stride=256):
        self.texts = texts
        self.labels = labels  # 多标签，如[[1,0,1,0],[0,1,1,1],...]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.processed_data = []
        self.process_examples()

    def process_examples(self):
        """
        将长文本切分为多个片段，并保持与标签的映射关系。
        在多标签场景下，可以采用"对整条文本的标签进行复制"或"仅对某片段标签加权"策略；
        这里演示的是对文本切片都视为相同标签，测评时再聚合。
        """
        for text, label in zip(self.texts, self.labels):
            # 使用encode_plus带滑窗分块(tokenize, encode)
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                stride=self.stride,
                return_overflowing_tokens=True,
                padding="max_length",
                return_tensors='pt'
            )

            for i in range(len(encoded['input_ids'])):
                chunk_input_ids = encoded['input_ids'][i]
                chunk_attention_mask = encoded['attention_mask'][i]
                chunk = {
                    'input_ids': chunk_input_ids,
                    'attention_mask': chunk_attention_mask,
                    'labels': torch.tensor(label, dtype=torch.float)
                }
                self.processed_data.append(chunk)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

# ------------------------------
# 2. 评估函数
# ------------------------------
def compute_metrics(pred):
    """
    计算多标签分类的评估指标
    """
    predictions = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5).numpy()
    labels = pred.label_ids
    
    # 计算每个标签的precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='micro'
    )
    
    # 计算准确率
    acc = accuracy_score(labels.flatten(), predictions.flatten())
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ------------------------------
# 3. 模型训练
# ------------------------------
def train_model():
    # 假设有12个标签，对应需求描述的12大类：
    # 公告通知、公告公文、功能说明书、学术论文、法律合同、技术文档、
    # 各类报告、年终总结、活动方案、需求文档、产品手册、数据报表、会议纪要
    # 这里仅举例12个标签（按需可扩展或修改）
    num_labels = 12

    # 检测是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1) 模型与分词器
    model_name = "modernbert-base-chinese"  # 或改用具体路径
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        torch_dtype=torch.float32  # 明确指定数据类型
    )
    
    # 将模型移到指定设备
    model = model.to(device)

    # 2) 构造训练和测试数据集
    # 假设 texts, labels 是事先准备好的
    train_texts = [
        "这是一份关于某次活动方案的文档，内容包含了活动时间、场地和预算等详细信息......",
        "本技术文档主要介绍系统的功能说明书，包括接口定义、模块设计和安全方案等..."
    ]
    train_labels = [
        [0,0,0,0,0,0,0,0,1,0,0,0],  # 对应"活动方案"
        [0,0,1,0,0,1,0,0,0,0,0,0]   # 对应"功能说明书" + "技术文档"
    ]

    test_texts = train_texts[:1]  # 示例用，实际应使用真实测试数据
    test_labels = train_labels[:1]

    train_dataset = LongTextDataset(train_texts, train_labels, tokenizer)
    test_dataset = LongTextDataset(test_texts, test_labels, tokenizer)

    # 3) 训练参数
    training_args = TrainingArguments(
        output_dir="./modernbert_multilabel",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        save_total_limit=2,
        # 如果有GPU则启用fp16
        fp16=torch.cuda.is_available(),
    )

    # 4) 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 5) 开始训练
    trainer.train()

    # 6) 保存模型
    trainer.save_model("./modernbert_multilabel")
    tokenizer.save_pretrained("./modernbert_multilabel")

# ------------------------------
# 4. 推理
# ------------------------------
def predict_long_text(model, tokenizer, text, max_length=512, stride=256, threshold=0.5):
    model.eval()

    # 对长文本进行同样的滑窗分块
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors='pt'
    )

    # 存储所有 chunk 的预测结果
    preds_list = []

    with torch.no_grad():
        for i in range(len(encoded['input_ids'])):
            input_ids = encoded['input_ids'][i].unsqueeze(0)
            attention_mask = encoded['attention_mask'][i].unsqueeze(0)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits.squeeze(0)
            sigmoid_out = torch.sigmoid(logits)
            preds_list.append(sigmoid_out)

    # 聚合，如取平均分
    aggregated = torch.mean(torch.stack(preds_list, dim=0), dim=0)
    # 多标签分类, 大于threshold则判断为该类
    result = (aggregated > threshold).int().cpu().numpy().tolist()
    return result


if __name__ == "__main__":
    # 1) 训练示例
    train_model()

    # 2) 推理示例
    model_path = "./modernbert_multilabel"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_text = "这是一个活动方案的文档，详细说明了会议纪要、活动内容与预算规划等内容……"

    # predict_result 类似 [0,0,0,0,0,0,0,0,1,0,0,1]，表明其对应活动方案(索引8)和会议纪要(索引11)
    predict_result = predict_long_text(model, tokenizer, test_text)
    print("推断结果:", predict_result)