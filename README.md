# vLLM 服务 - 支持 Logits 提取和 Reliability 计算

## 新增功能

### 1. ✅ 多模态支持 (Qwen2.5-VL)
- 支持图像+文本输入
- 自动检测多模态模型
- 最多支持10张图片

### 2. ✅ 整句 Reliability 计算 (LogTokU 方法)
- 基于论文方法,使用 EU×AU 计算每个token的不确定性
- 每个token使用 top-2 logits 计算
- 支持选择最不确定的K个tokens来代表整句可靠性

## API 使用

### 基本用法

```bash
# 启动服务
python deploy_vllm_with_logits.py

# 或指定多模态模型
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct" python deploy_vllm_with_logits.py
```

### API 请求示例

#### 1. 纯文本 + Reliability 计算

```python
import requests

response = requests.post("http://localhost:8000/v1/completions_with_logits", json={
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "temperature": 0.7,
    "top_k_logits": 5,
    "calculate_reliability": True,      # 启用 Reliability 计算
    "reliability_k_tokens": 5,          # 使用最不确定的5个tokens
})

result = response.json()

# 输出结果
print("生成文本:", result['text'])
print("整句可靠性:", result['reliability']['sentence_reliability'])
print("平均不确定性:", result['reliability']['avg_uncertainty'])
```

#### 2. 多模态输入 (Qwen2.5-VL)

```python
response = requests.post("http://localhost:8000/v1/completions_with_logits", json={
    "prompt": {
        "prompt": "Describe this image in detail:",
        "images": ["https://example.com/image.jpg"]
    },
    "max_tokens": 50,
    "calculate_reliability": True,
})

result = response.json()
print("描述:", result['text'])
print("可靠性:", result['reliability']['sentence_reliability'])
```

### 响应格式

```json
{
  "text": " Paris",
  "token_ids": [12095],
  "logits": [
    {
      "step": 0,
      "top_k": [...],
      "vocab_size": 151936,
      "eu_2": 0.0565,
      "au_2": 0.6780,
      "uncertainty": 0.038335
    }
  ],
  "num_steps": 1,
  "reliability": {
    "method": "LogTokU (EU×AU based on top-2 logits)",
    "avg_all_tokens": 0.038335,
    "avg_uncertainty": 0.038335,
    "top_5_uncertain_tokens": 0.038335,
    "sentence_reliability": -0.038335,
    "interpretation": {
      "avg_uncertainty": "平均不确定性 (所有tokens的EU×AU平均值)",
      "top_k_uncertainty": "最不确定的5个tokens的EU×AU平均值",
      "sentence_reliability": "整句可靠性 (越接近0越可靠, 越负越不可靠)"
    },
    "token_level": {
      "eu_2": [0.0565],
      "au_2": [0.6780],
      "uncertainty": [0.038335]
    }
  }
}
```

## 参数说明

### LogitsRequest 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str \| dict | 必需 | 文本提示或多模态输入 |
| `max_tokens` | int | 100 | 最大生成token数 |
| `temperature` | float | 0.7 | 采样温度 |
| `top_p` | float | 0.95 | Top-p采样 |
| `top_k_logits` | int | 10 | 返回每个token的top-k logits |
| `return_full_logits` | bool | False | 是否返回完整logits向量 |
| `calculate_reliability` | bool | True | 是否计算 Reliability |
| `reliability_k_tokens` | int | 5 | 选择最不确定的K个tokens |
| `images` | List[str] | None | 图像URL列表(多模态) |

## Reliability 指标解读

### 指标说明

1. **EU (Expected Utility)**: 期望效用
   - 基于top-2 logits计算
   - 范围: (0, 1]
   - 值越大表示不确定性越高

2. **AU (Aleatoric Uncertainty)**: 任意不确定性
   - 基于top-2 logits计算  
   - 范围: (0, 1]
   - 值越大表示不确定性越高

3. **EU×AU**: 综合不确定性
   - 两者相乘
   - 范围: (0, 1)
   - 值越大表示token越不确定

4. **Sentence Reliability**: 整句可靠性
   - 计算公式: `Reliability = -mean(EU×AU of top-K uncertain tokens)`
   - **值越接近0 = 模型越有信心**
   - **值越负 = 模型越不确定**

### 使用建议

- **高可靠性场景** (Reliability > -0.03): 可以直接使用生成结果
- **中等可靠性** (-0.05 < Reliability < -0.03): 建议人工审核
- **低可靠性** (Reliability < -0.05): 需要人工干预或重新生成

## 论文方法

本实现基于 **LogTokU** 论文方法:

- 使用 top-2 logits 计算每个token的 EU 和 AU
- 选择最不确定的K个tokens代表整句可靠性
- 自动过滤掉不重要的tokens(如标点符号)

参考文献:
```
LogTokU: A Token-Level Uncertainty Quantification Method for Large Language Models
```


## 环境变量

```bash
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"  # 或 Qwen/Qwen2.5-VL-7B-Instruct
HOST="0.0.0.0"
PORT="8000"
TENSOR_PARALLEL_SIZE="1"
MAX_MODEL_LEN="4096"
GPU_MEMORY_UTIL="0.85"
ENFORCE_EAGER="True"
```

## 依赖

```bash
pip install vllm fastapi uvicorn torch numpy scipy
```

git remote add origin git@github.com:zhhvvv/output_with_logits.git