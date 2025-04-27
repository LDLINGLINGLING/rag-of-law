
# RAG_Of_Law: 基于RAFT框架的法律问答系统

### 项目概述
本项目采用RAFT(Retrieval-Augmented Fine-Tuning)技术框架，构建面向法律领域的专业问答系统。通过结合检索增强生成(RAG)和监督微调(SFT)，显著提升模型在复杂法律问题上的回答准确性和专业性。

GitHub地址: [https://github.com/LDLINGLINGLING/rag-of-law](https://github.com/LDLINGLINGLING/rag-of-law)

### 技术架构
用户提问 → 分类模型 → 法律类别判定 → 语义检索 → LLM生成 → 专业回答


### 核心组件

#### 1. 分类模型
- 使用BERT-base架构
- 两级分类体系：
  - 12个法律大类（准确率98.5%）
  - 115个法律子类（准确率97.6%）

#### 2. 语义检索模型
- 对比学习训练
- BM25混合检索

#### 3. 生成模型
- 抗噪声输入设计
- 专业表述重构
- 支持多轮对话

### 数据集

#### 法律条款库
```json
[
  ["民法", "合同法", "第五百条 当事人应当遵循诚信原则..."],
  ["刑法", "刑法修正案(十一)", "三十二、将刑法第一百四十一条修改为..."]
]
```
#### 法律数据集
```json
{
  "question": "公司未制止性骚扰的责任",
  "answer": "根据《妇女权益保障法》第八十条...",
  "references": [
    "第七十九条 违反报告义务的处罚",
    "第八十条 性骚扰处置责任"
  ]
}
```
## 构建具有特定风格的聊天机器人
本项目在于构建垂直领域的聊天机器人
###  项目目标：
1. 构造一个大学生心理咨询机器人。
2. 能够耐心、热情、富有同情心、感染力的回复，数据需要带有这种风格（可以选择豆包进行蒸馏）。
3. 能够自主提问，并非简单的回答问题（在构造数据时需要提出这个要求）。
4. 能够根据学生的个人背景信息个性化回复（在训练数据的时候需要加入个人信息）。
5. 能够陪学生侃大山，天南地北的聊天，无bug（泛化性要好，不能有恶劣badcase）
### 技术路线：
1. 收集学生常见聊天场景（可大模型生成，需要多样性，保证模型训练的泛化性）
2. 总结学生用户背景如：学历、年纪、专业、性格、学校等等（可以大模型生成）
3. 收集真实的心理咨询师与大学生的对话数据集

具体效果如下：
<img width="1280" alt="image" src="https://github.com/LDLINGLINGLING/rag-of-law/assets/47373076/31317e06-ff16-4daf-a440-ef0dff186c19">

<img width="1280" alt="image" src="https://github.com/LDLINGLINGLING/rag-of-law/assets/47373076/702fe885-57ca-4ebf-80fe-c18c147549aa">

<img width="1280" alt="image" src="https://github.com/LDLINGLINGLING/rag-of-law/assets/47373076/4f5534e7-1f39-4f61-97e7-c84751fca9ae">

<img width="1280" alt="image" src="https://github.com/LDLINGLINGLING/rag-of-law/assets/47373076/3c13cd5e-95af-45e1-9d38-2134dc115c86">

<img width="1280" alt="image" src="https://github.com/LDLINGLINGLING/rag-of-law/assets/47373076/8c50a74e-0d04-4cac-918c-146f8f87e13b">

<img width="1280" alt="image" src="https://github.com/LDLINGLINGLING/rag-of-law/assets/47373076/d28bd824-c184-41cb-b729-f0399a68449b">



