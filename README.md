# SmallVille: Resource Scramble Experiment

**语言融合实验** - 研究零和竞争环境下皮钦语(Pidgin)的产生

## 核心研究问题

在零和博弈中,当两个团队必须通过**强制的抽象符号**沟通时:
- L_A 和 L_B 是否会融合形成共享的皮钦语？
- 智能体是否会发展双语系统(团队内部语言 vs. 跨团队语言)？
- 符号的语义如何从无意义逐步稳定？

## 实验设计

### 环境: Resource Scramble (资源争夺)

```
Team A (位置持有者)          Team B (属性持有者)
    ↓                              ↓
知道资源坐标 (x, y)         知道资源属性 (颜色、形状、价值、陷阱)
    ↓                              ↓
        必须通过抽象符号沟通
    ↓                              ↓
零和竞争: 最先找到最佳资源的团队获胜
```

### 核心机制

#### 1. 强制抽象词表 (Forced Abstract Vocabulary)
- **词表**: `tok1, tok2, ..., tok20` (无预设语义)
- **约束**: 
  - 只能使用这20个符号
  - 最大消息长度: 5个token
  - 使用自然语言 → 立即拒绝 + 惩罚

#### 2. 课程学习 (Curriculum Learning)
- **Phase 1**: 4个资源, 2个特征维度
- **Phase 2**: 6个资源, 2个特征维度  
- **Phase 3**: 8个资源, 2个特征维度
- **晋级条件**: 成功率 > 80%

#### 3. 信息不对称 + 零和博弈
```python
Team A 知道: {位置}
Team B 知道: {颜色, 形状, 价值, 是否陷阱}

目标: 找到最高价值且非陷阱的资源
竞争: 团队A获胜 = +R, 团队B失败 = -R
```

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 设置 DeepSeek API Key

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key-here"
```

### 3. 运行实验

```powershell
python run_experiment.py
```

## 实验输出

运行后会在 `results/exp_YYYYMMDD_HHMMSS/` 生成:

### 1. `metrics.json` - 语言指标
```json
{
  "cross_team_mi": 0.234,           // 跨团队互信息
  "compositionality": 0.567,         // 组合性得分
  "symbol_reuse": 0.823,             // 符号复用率
  "shared_vocab_size": 8,            // 共享词汇量
  "team_a_unique": 5,                // A队独有符号
  "team_b_unique": 7,                // B队独有符号
  "efficiency": 0.145                // 通信效率
}
```

### 2. `pidgin_analysis.json` - 皮钦语检测
```json
{
  "detected": true,
  "stable_pidgin_tokens": ["tok3", "tok7", "tok11"],
  "pidgin_semantics": {
    "tok3": [["color:red", 0.87], ["value:high", 0.65]],
    "tok7": [["shape:circle", 0.92]],
    "tok11": [["is_trap:false", 0.78]]
  }
}
```

### 3. `communications.txt` - 对话记录
```
=== Episode 45 ===
Round 1:
  A: tok3, tok7 -> 2
  B: tok11, tok15 -> 2
  Winner: A

Round 2:
  A: tok3, tok11, tok8 -> 1
  B: tok7, tok3 -> 1
  Winner: tie
```

### 4. `token_semantics.json` - 符号语义演化
显示每个token与概念的关联强度

## 实验参数调整

编辑 `run_experiment.py`:

```python
config = {
    'vocabulary_size': 20,      # 词表大小 K
    'max_message_length': 5,    # 消息长度 L
    'max_rounds': 10,           # 每episode最大轮数
    'num_episodes': 100,        # 总episode数
}
```

## 关键指标解读

### 跨团队互信息 (Cross-team MI)
- **含义**: I(Message_A ; Outcome_B) - A的消息对B成功的预测能力
- **高值**: 共享理解形成
- **低值**: 沟通混乱

### 组合性 (Compositionality)
- **含义**: 符号是否系统性组合 (e.g., tok3+tok8 = "红圆")
- **高值**: 出现组合语法
- **低值**: 整体性编码

### 符号复用率 (Symbol Reuse)
- **含义**: 多少符号在多个语境中重复使用
- **高值**: 抽象、高效的语言
- **低值**: 专用符号(低效)

### 皮钦语检测
检测标准:
1. ≥3个符号被两队稳定共用
2. 最近10轮消息中至少出现5次
3. 语义关联明确

## 架构说明

```
run_experiment.py          # 主入口
    ↓
simulation.py              # 实验编排器
    ↓
├── resource_scramble.py   # 游戏环境
│   ├── ResourceScrambleEnvironment  (资源生成、奖励计算)
│   ├── AbstractVocabulary           (词表验证)
│   └── CurriculumManager            (课程学习)
│
├── llm_backend.py         # LLM后端
│   ├── DeepSeekBackend              (API调用)
│   ├── PromptBuilder                (提示词构建)
│   └── MessageParser                (响应解析)
│
└── language_analysis.py   # 语言分析
    ├── LanguageAnalyzer             (指标计算)
    ├── SymbolSemantics              (语义追踪)
    └── PidginDetector               (皮钦检测)
```

## 研究意义

### 理论价值
1. **验证零和环境下的语言融合假设**
2. **观察组合性语法的自然产生**
3. **理解信息不对称对语言结构的影响**

### 方法创新
1. **强制抽象通道** - 完全阻断预训练语言
2. **双重压力** - 合作需求 + 竞争约束
3. **课程学习** - 自然难度递增

## 预期观察

### 早期 (Episodes 1-20)
- 随机探索
- 低成功率 (~30%)
- 无明显符号语义

### 中期 (Episodes 20-60)
- 组合性出现 (tok3 = 颜色, tok7 = 形状)
- 成功率上升 (~60%)
- 部分符号语义稳定

### 后期 (Episodes 60-100)
- **皮钦语形成**: 3-5个核心共享符号
- 高效沟通 (成功率 > 80%)
- 符号语义收敛

## 扩展方向

1. **增加团队数**: 3个或更多团队
2. **动态词表**: 允许创造新符号
3. **噪声信道**: 添加通信错误
4. **代际传递**: 新agent替换旧agent

## 引用

基于以下研究设计:
- Emergent Communication (Lewis et al., 2017)
- Referential Games (Lazaridou et al., 2020)
- SmallVille Architecture (Park et al., 2023)

## License

MIT License
