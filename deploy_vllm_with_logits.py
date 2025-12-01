"""
vLLM服务 - 支持OpenAI API和原始logits提取
支持多模态模型和置信度计算
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
import uvicorn
import torch
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# 导入置信度计算模块
from confidence import calculate_confidence, aggregate_sentence_confidence

os.environ["VLLM_USE_V1"] = "0"

app = FastAPI(title="vLLM with Logits")

# 配置
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "1024"))
GPU_MEMORY_UTIL = float(os.getenv("GPU_MEMORY_UTIL", "0.65"))
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "True").lower() == "true"

llm = None

def download_image(url: str) -> Image.Image:
    """从URL下载图像并返回PIL Image对象"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise HTTPException(400, f"Failed to download image from {url}: {str(e)}")

class LogitsSpy:
    def __init__(self):
        self.processed_logits: List[torch.Tensor] = []
    
    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        # 转换为float32以避免BFloat16兼容性问题
        self.processed_logits.append(logits.detach().cpu().float())
        return logits

class ChatMessage(BaseModel):
    role: str
    content: str | List[Dict[str, Any]]  # 支持多模态内容

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    images: Optional[List[str]] = None  # 可选的图像列表

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

class LogitsRequest(BaseModel):
    prompt: str | Dict[str, Any]  # 支持文本或多模态输入
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95
    top_k_logits: int = Field(default=10, description="返回top-k，-1返回全部")
    return_full_logits: bool = Field(default=False, description="返回完整logits向量")
    calculate_reliability: bool = Field(default=True, description="计算整句Reliability")
    reliability_k_tokens: int = Field(default=5, description="计算Reliability时使用的top-k tokens")
    images: Optional[List[str]] = None  # 可选的图像URL列表
    image_url: Optional[str] = None  # 兼容单个图像URL

@app.on_event("startup")
async def startup():
    global llm
    print("=" * 80)
    print(f"🚀 Loading {MODEL_NAME}")
    print(f"📍 {HOST}:{PORT}")
    
    # 检测是否是多模态模型
    is_multimodal = "vl" in MODEL_NAME.lower() or "vision" in MODEL_NAME.lower()
    if is_multimodal:
        print("🖼️  Multimodal model detected")
    
    print("=" * 80)
    
    llm_kwargs = {
        "model": MODEL_NAME,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTIL,
        "trust_remote_code": True,
        "enforce_eager": ENFORCE_EAGER,
    }
    
    # 多模态模型特殊配置
    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 4}  # 支持最多4张图片
    print(llm_kwargs)
    llm = LLM(**llm_kwargs)
    print("✅ Ready!")
    print(f"✅ Confidence calculation enabled (LogTokU method)")
    if is_multimodal:
        print(f"✅ Multimodal support enabled")

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")
    
    prompt = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
    prompt += "\nassistant:"
    
    outputs = llm.generate([prompt], SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature
    ))
    
    return {
        "id": "chat-" + os.urandom(4).hex(),
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "message": {"role": "assistant", "content": outputs[0].outputs[0].text},
            "finish_reason": "stop"
        }]
    }

@app.post("/v1/completions")
async def completion(request: CompletionRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")
    
    outputs = llm.generate([request.prompt], SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature
    ))
    
    return {
        "id": "cmpl-" + os.urandom(4).hex(),
        "object": "text_completion",
        "model": request.model,
        "choices": [{"text": outputs[0].outputs[0].text, "finish_reason": "stop"}]
    }

@app.post("/v1/completions_with_logits")
async def completion_with_logits(request: LogitsRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")
    
    try:
        logits_spy = LogitsSpy()
        
        # 处理输入 (支持文本和多模态)
        if isinstance(request.prompt, dict):
            # 多模态输入
            prompt_input = request.prompt
        else:
            # 纯文本输入
            text_prompt = request.prompt
            
            # 如果提供了image_url或images,构建多模态输入
            if request.image_url or request.images:
                image_urls = []
                if request.image_url:
                    image_urls.append(request.image_url)
                if request.images:
                    image_urls.extend(request.images)
                
                # 下载图像并转换为PIL Image对象
                pil_images = [download_image(url) for url in image_urls]
                
                # 为Qwen2-VL添加图像占位符
                # 每张图片需要一个 <|image_pad|> 占位符
                image_placeholders = "<|image_pad|>" * len(pil_images)
                prompt_with_placeholder = f"{image_placeholders}{text_prompt}"
                
                # 构建vLLM的多模态输入格式
                # 对于单张图片,直接传递Image对象;多张图片传递列表
                prompt_input = {
                    "prompt": prompt_with_placeholder,
                    "multi_modal_data": {
                        "image": pil_images[0] if len(pil_images) == 1 else pil_images
                    }
                }
            else:
                prompt_input = text_prompt
        
        outputs = llm.generate([prompt_input], SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            logits_processors=[logits_spy]
        ))
        
        output = outputs[0].outputs[0]
        result_logits = []
        
        # 用于计算 reliability 的数据
        eu_2_scores = []
        au_2_scores = []
        
        for step_idx, step_logits in enumerate(logits_spy.processed_logits):
            # 处理形状
            if len(step_logits.shape) == 2:
                logits_tensor = step_logits[0]
            elif len(step_logits.shape) == 3:
                logits_tensor = step_logits[0, 0]
            else:
                logits_tensor = step_logits
            
            vocab_size = logits_tensor.shape[0]
            
            # 计算置信度指标 (使用 top-2 logits)
            if request.calculate_reliability:
                logits_np = logits_tensor.cpu().numpy()
                eu_2 = calculate_confidence(logits_np, mode="eu_2")
                au_2 = calculate_confidence(logits_np, mode="au_2")
                eu_2_scores.append(float(eu_2))
                au_2_scores.append(float(au_2))
            
            if request.return_full_logits:
                step_data = {
                    "step": step_idx,
                    "full_logits": logits_tensor.tolist(),
                    "vocab_size": vocab_size
                }
            else:
                k = vocab_size if request.top_k_logits == -1 else min(request.top_k_logits, vocab_size)
                top_values, top_indices = torch.topk(logits_tensor, k)
                
                step_data = {
                    "step": step_idx,
                    "top_k": [
                        {"token_id": int(idx), "logit": float(val), "rank": i + 1}
                        for i, (val, idx) in enumerate(zip(top_values, top_indices))
                    ],
                    "vocab_size": vocab_size
                }
            
            # 添加置信度信息到每个step
            if request.calculate_reliability:
                step_data["eu_2"] = eu_2_scores[-1]
                step_data["au_2"] = au_2_scores[-1]
                step_data["uncertainty"] = eu_2_scores[-1] * au_2_scores[-1]  # EU×AU
            
            result_logits.append(step_data)
        
        # 计算整句的 Reliability
        response = {
            "text": output.text,
            "token_ids": output.token_ids,
            "logits": result_logits,
            "num_steps": len(result_logits)
        }
        
        if request.calculate_reliability and len(eu_2_scores) > 0:
            combined = np.array(eu_2_scores) * np.array(au_2_scores)
            
            # 整句可靠性指标
            avg_reliability = float(np.mean(combined))
            
            # Top-K 最不确定的tokens (论文方法)
            k_tokens = min(request.reliability_k_tokens, len(combined))
            if k_tokens == len(combined):
                topk_reliability = avg_reliability
            else:
                top_k_indices = np.argpartition(combined, -k_tokens)[-k_tokens:]
                topk_reliability = float(np.mean(combined[top_k_indices]))
            
            response["reliability"] = {
                "method": "LogTokU (EU×AU based on top-2 logits)",
                "avg_all_tokens": avg_reliability,
                "avg_uncertainty": avg_reliability,  # 别名
                f"top_{k_tokens}_uncertain_tokens": topk_reliability,
                "sentence_reliability": -topk_reliability,  # 论文定义: Reliability = -AU×EU
                "interpretation": {
                    "avg_uncertainty": "平均不确定性 (所有tokens的EU×AU平均值)",
                    "top_k_uncertainty": f"最不确定的{k_tokens}个tokens的EU×AU平均值",
                    "sentence_reliability": "整句可靠性 (越接近0越可靠, 越负越不可靠)"
                },
                "token_level": {
                    "eu_2": eu_2_scores,
                    "au_2": au_2_scores,
                    "uncertainty": combined.tolist()
                }
            }
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}

@app.get("/health")
async def health():
    return {"status": "healthy" if llm else "not_ready", "model": MODEL_NAME}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)