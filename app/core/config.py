from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import os

class Settings(BaseSettings):
    # API Settings
    api_title: str = "Prompt Evaluation API"
    api_version: str = "0.1.0"
    
    # AWS Settings
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    
    # Model Configuration
    default_models: Dict[str, str] = {
        "type_a": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "type_b_text": "anthropic.claude-3-haiku-20240307-v1:0",
        "type_b_image": "amazon.nova-canvas-v1:0"
    }
    
    # Embedding Models
    embedding_models: Dict[str, str] = {
        # 텍스트 임베딩
        "titan_text": "amazon.titan-embed-text-v1",
        "cohere_multilingual": "cohere.embed-multilingual-v3",
        # 이미지/멀티모달 임베딩
        "titan_multimodal": "amazon.titan-embed-image-v1",
        "cohere_v4": "cohere.embed-v4"
    }
    
    # Judge Model (저렴한 모델)
    judge_model: str = "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Scoring Weights
    weights: Dict[str, Dict[str, float]] = {
        "type_a": {
            "token_usage": 0.15,
            "information_density": 0.20,
            "consistency": 0.20,
            "model_variance": 0.15,
            "hallucination": 0.15,
            "relevance": 0.15
        },
        "type_b_text": {
            "token_usage": 0.25,
            "information_density": 0.25,
            "model_variance": 0.25,
            "relevance": 0.25
        },
        "type_b_image": {
            "token_usage": 0.30,
            "consistency": 0.30,
            "model_variance": 0.20,
            "relevance": 0.20
        }
    }
    
    # Consistency Parameters
    alpha: float = 0.2  # centroid 계산에서 max penalty 가중치
    
    # Information Density Weights
    density_weights: Dict[str, float] = {
        "unigram": 0.5,
        "bigram": 0.5
    }
    
    # Database
    database_url: str = "sqlite:///./prompt_eval.db"
    
    # Storage Backend
    storage_backend: str = "sqlite"  # "sqlite" (기본) or "s3" or "dynamodb_s3"
    s3_bucket_name: str = "prompt-eval-bucket"
    table_name: str = "prompt-evaluations"
    
    # Cache Settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Mock Mode (테스트용)
    mock_mode: bool = False  # AWS 없이 테스트할 때 True
    
    # MCP Settings
    mcp_enabled: bool = True  # MCP 서버 사용 여부
    mcp_timeout: int = 30  # MCP 요청 타임아웃 (초)
    arxiv_mcp_path: str = ""  # ArXiv MCP 실행 파일 경로 (Windows .exe)
    
    class Config:
        env_file = ".env"

settings = Settings()