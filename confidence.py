"""
置信度计算库
"""
import numpy as np
from scipy.special import softmax, digamma

def topk(arr, k):
    indices = np.argpartition(arr, -k)[-k:]
    values = arr[indices]
    return values, indices

def calculate_confidence(logits, mode="prob", k=None):
    """
    计算单步置信度
    
    Args:
        logits: 原始logits向量（numpy array）
        mode: 'prob', 'entropy', 'eu', 'au'
        k: top-k参数（用于eu和au）
    
    Returns:
        float: 置信度分数
    """
    if mode == "prob":
        probs = softmax(logits)
        return np.max(probs)
    
    elif mode == "entropy":
        probs = softmax(logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    elif mode == "eu":
        if k is None:
            raise ValueError("k required for eu mode")
        if len(logits) < k:
            raise ValueError(f"logits length {len(logits)} < k {k}")
        top_values, _ = topk(logits, k)
        return k / (np.sum(np.maximum(0, top_values)) + k)
    
    elif mode == "au":
        if k is None:
            raise ValueError("k required for au mode")
        if len(logits) < k:
            raise ValueError(f"logits length {len(logits)} < k {k}")
        top_values = np.partition(logits, -k)[-k:]
        alpha = np.array([top_values])
        alpha_0 = alpha.sum(axis=1, keepdims=True)
        psi_alpha_k_plus_1 = digamma(alpha + 1)
        psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
        result = -(alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
        return result.sum(axis=1)[0]
    
    elif mode == "eu_2":
        # 固定使用top-2的logits计算EU
        top_k = 2
        if len(logits) < top_k:
            raise ValueError(f"logits length {len(logits)} < top_k {top_k}")
        top_values, _ = topk(logits, top_k)
        return top_k / (np.sum(np.maximum(0, top_values)) + top_k)
    
    elif mode == "au_2":
        # 固定使用top-2的logits计算AU
        top_k = 2
        if len(logits) < top_k:
            raise ValueError(f"logits length {len(logits)} < top_k {top_k}")
        top_values = np.partition(logits, -top_k)[-top_k:]
        alpha = np.array([top_values])
        alpha_0 = alpha.sum(axis=1, keepdims=True)
        psi_alpha_k_plus_1 = digamma(alpha + 1)
        psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
        result = -(alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
        return result.sum(axis=1)[0]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def calculate_all_steps(logits_data, mode="prob", k=None):
    """
    计算所有步骤的置信度
    
    Args:
        logits_data: API返回的logits数据
        mode: 置信度模式
        k: top-k参数
    
    Returns:
        list: 每步的置信度
    """
    scores = []
    for step_data in logits_data:
        if "full_logits" in step_data:
            logits = np.array(step_data["full_logits"])
        else:
            # 从top_k重建
            vocab_size = step_data["vocab_size"]
            logits = np.full(vocab_size, -100.0)
            for item in step_data["top_k"]:
                logits[item["token_id"]] = item["logit"]
        
        score = calculate_confidence(logits, mode=mode, k=k)
        scores.append(score)
    
    return scores

def aggregate_sentence_confidence(logits_data, method="avg", k_tokens=None):
    """
    计算整句话的置信度/可靠性分数
    
    Returns:
        float: 可靠性分数 (越大越可靠)
    """
    if k_tokens is None:
        k_tokens = 25
    
    if method == "avg_logtu":
        eu_scores = calculate_all_steps(logits_data, mode="eu_2")
        au_scores = calculate_all_steps(logits_data, mode="au_2")
        combined = np.array(eu_scores) * np.array(au_scores)
        return -np.mean(combined)  # 负号：不确定性 → 可靠性
    
    elif method == "topk_logtu":
        eu_scores = calculate_all_steps(logits_data, mode="eu_2")
        au_scores = calculate_all_steps(logits_data, mode="au_2")
        combined = np.array(eu_scores) * np.array(au_scores)
        
        k = min(k_tokens, len(combined))
        if k == len(combined):
            return -np.mean(combined)
        
        # 选择最大的k个 (最不确定的tokens)
        top_k_indices = np.argpartition(combined, -k)[-k:]
        return -np.mean(combined[top_k_indices])  # 负号：不确定性 → 可靠性
    
    elif method == "avg_prob":
        prob_scores = calculate_all_steps(logits_data, mode="prob")
        # log(prob): 概率越高越接近0，越低越负
        return -np.mean(-np.log(np.array(prob_scores) + 1e-10))
    
    elif method == "topk_prob":
        prob_scores = calculate_all_steps(logits_data, mode="prob")
        k = min(k_tokens, len(prob_scores))
        probs = np.array(prob_scores)
        
        # -log(prob) 是不确定性，选最大的k个
        uncertainty = -np.log(probs + 1e-10)
        if k == len(uncertainty):
            return -np.mean(uncertainty)
        
        top_k_indices = np.argpartition(uncertainty, -k)[-k:]
        return -np.mean(uncertainty[top_k_indices])
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")