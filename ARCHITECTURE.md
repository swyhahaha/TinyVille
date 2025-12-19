# 实验流程图 (Experiment Flow)

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   SmallVille Simulation                      │
│                    (simulation.py)                           │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Environment │ │  LLM Backend │ │   Language   │
│  (resource_  │ │  (llm_       │ │   Analyzer   │
│  scramble)   │ │  backend)    │ │  (language_  │
│              │ │              │ │  analysis)   │
└──────────────┘ └──────────────┘ └──────────────┘
```

## 单个Episode流程

```
Episode开始
    │
    ├─> Environment.reset()
    │   └─> 生成N个随机资源
    │       (位置、颜色、形状、价值、陷阱)
    │
    ├─> 获取分割视图
    │   ├─> Team A View: [位置信息]
    │   └─> Team B View: [属性信息]
    │
    └─> Round循环 (最多max_rounds轮)
        │
        ├─> Team A Turn
        │   ├─> 构建Prompt (location_data + 对手消息)
        │   ├─> 查询DeepSeek API
        │   ├─> 解析响应
        │   │   ├─> 提取消息 (只能用tok1-tok20)
        │   │   └─> 提取选择 (resource_id)
        │   └─> 验证: 消息合法? → 否则惩罚并重试
        │
        ├─> Team B Turn
        │   ├─> 构建Prompt (attribute_data + 对手消息)
        │   ├─> 查询DeepSeek API
        │   ├─> 解析响应
        │   └─> 验证消息合法性
        │
        ├─> 评估选择
        │   ├─> Team A: evaluate_choice() → (reward_a, correct_a)
        │   ├─> Team B: evaluate_choice() → (reward_b, correct_b)
        │   └─> 零和调整:
        │       if A正确 and B错误: A+10, B-10
        │       if B正确 and A错误: B+10, A-10
        │
        ├─> 记录轮次数据
        │   ├─> GameRound(messages, choices, rewards, winner)
        │   └─> 更新会话历史
        │
        ├─> 语言分析记录
        │   └─> LanguageAnalyzer.record_communication()
        │
        └─> 检查收敛
            └─> 有人找到最优? → break
```

## 课程学习流程

```
Phase 1 (简单)
├─> N=4资源, 2特征维度
├─> 运行episodes直到:
│   ├─> 成功率 > 80%
│   └─> 至少20个episodes
└─> ✓ 晋级 → Phase 2

Phase 2 (中等)
├─> N=6资源, 2特征维度
├─> 运行episodes直到达标
└─> ✓ 晋级 → Phase 3

Phase 3 (困难)
├─> N=8资源, 2特征维度
└─> 运行至实验结束
```

## 语言分析管道

```
每次通信后:
    │
    ├─> 记录消息对
    │   ├─> Team A: "tok3, tok7"
    │   └─> Team B: "tok11, tok15"
    │
    ├─> 记录结果
    │   ├─> success/fail
    │   └─> 资源属性
    │
    └─> 更新语义追踪
        └─> tok3 → {color:red: 0.87, value:high: 0.65}

实验结束时:
    │
    ├─> 计算互信息
    │   └─> I(Message_A ; Outcome_B)
    │
    ├─> 计算组合性
    │   └─> 位置一致性分析
    │
    ├─> 检测皮钦语
    │   ├─> 找共享符号
    │   ├─> 检查稳定性
    │   └─> 提取语义
    │
    └─> 生成报告
```

## 强制抽象通道工作原理

```
Agent想说: "Resource 2 is high value and safe"
    │
    ├─> LLM生成响应
    │
    ├─> MessageParser验证
    │   │
    │   ├─> 检查格式: "MESSAGE: ... CHOICE: ..."
    │   │
    │   ├─> 提取消息部分
    │   │
    │   ├─> 分割tokens: ["tok3", "tok7", "tok11"]
    │   │
    │   └─> 验证每个token
    │       ├─> "tok3" ✓ 在词表中
    │       ├─> "tok7" ✓ 在词表中
    │       ├─> "high" ✗ 不在词表中!
    │       └─> → 拒绝消息, 惩罚, 要求重试
    │
    └─> 如果全部通过 → 接受消息
        └─> 对手收到: "tok3, tok7, tok11"
            (必须自己理解含义!)
```

## 数据流

```
┌─────────────┐
│  DeepSeek   │
│     API     │
└──────┬──────┘
       │ response
       ▼
┌─────────────┐     valid message     ┌─────────────┐
│   Message   │ ───────────────────> │ Environment │
│   Parser    │                       │  Evaluator  │
└─────────────┘                       └──────┬──────┘
       │                                     │
       │ tokens + context                   │ rewards
       ▼                                     ▼
┌─────────────┐                       ┌─────────────┐
│  Language   │ ←──── feedback ────── │  Episode    │
│  Analyzer   │                       │  Result     │
└─────────────┘                       └─────────────┘
```

## 关键时刻

### 1. 语言创造时刻
```
Round 1: Team A随机发送 "tok5, tok12"
         Team B无法理解 → 随机猜测 → 失败

Round 5: Team A开始关联 tok5 = "位置靠近"
         Team B注意到tok5出现时A常选某区域

Round 15: 双方隐式达成: tok5 ≈ "上半区域"
          tok12 ≈ "高价值"
```

### 2. 组合性出现
```
Phase 1后期:
    tok3 单独出现 → "红色"
    tok8 单独出现 → "圆形"
    
Phase 2:
    tok3, tok8 一起 → "红圆形" (组合!)
```

### 3. 皮钦稳定
```
Episode 60+:
    核心pidgin词汇固定:
    - tok3: 红色 (p=0.92)
    - tok7: 圆形 (p=0.89)
    - tok11: 安全 (p=0.85)
    - tok15: 高价值 (p=0.78)
```
