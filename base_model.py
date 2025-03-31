import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

snapshot_download(
    'LLM-Research/Meta-Llama-3.1-8B-Instruct', 
    cache_dir='base_model/llama3.1-8b-instruct',
    revision='master'
)