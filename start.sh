#!/bin/bash
export $(cat config.env | xargs)
export CUDA_VISIBLE_DEVICES=3,7
python deploy_vllm_with_logits.py