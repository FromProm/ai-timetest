import json
import logging
import base64
import boto3
from typing import List, Optional
from app.adapters.embedder.base import BaseEmbedder
from app.core.config import settings
from app.core.errors import EmbeddingError

logger = logging.getLogger(__name__)

class BedrockEmbedder(BaseEmbedder):
    """AWS Bedrock 임베딩 생성기"""
    
    def __init__(self):
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None
        )
    
    # ========== 텍스트 임베딩 ==========
    
    async def embed_text(self, text: str) -> List[float]:
        """Titan Text 임베딩"""
        return await self._invoke_embedding(
            settings.embedding_models['titan_text'],
            {"inputText": text}
        )
    
    async def embed_multilingual(self, text: str) -> List[float]:
        """Cohere Multilingual 임베딩 (텍스트용)"""
        return await self._invoke_embedding(
            settings.embedding_models['cohere_multilingual'],
            {
                "texts": [text],
                "input_type": "search_document"
            }
        )
    
    # ========== 이미지 임베딩 ==========
    
    async def embed_image_titan(self, image_base64: str, text: Optional[str] = None) -> List[float]:
        """
        Titan Multimodal 이미지 임베딩
        - image_base64: base64 인코딩된 이미지
        - text: 선택적 텍스트 (이미지와 함께 임베딩)
        """
        body = {"inputImage": image_base64}
        if text:
            body["inputText"] = text
        
        return await self._invoke_embedding(
            settings.embedding_models['titan_multimodal'],
            body
        )
    
    async def embed_image_cohere(self, image_base64: str) -> List[float]:
        """
        Cohere Embed v4 이미지 임베딩
        - image_base64: base64 인코딩된 이미지
        """
        # Cohere v4는 data URI 형식 필요
        # 이미지 타입 감지 (간단히 PNG/JPEG 가정)
        if image_base64.startswith('/9j/'):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/png'
        
        image_uri = f"data:{mime_type};base64,{image_base64}"
        
        body = {
            "input_type": "image",
            "inputs": [
                {
                    "content": [
                        {"type": "image_url", "image_url": image_uri}
                    ]
                }
            ],
            "embedding_types": ["float"]
        }
        
        return await self._invoke_cohere_v4_embedding(body)
    
    async def embed_image_with_text_cohere(self, image_base64: str, text: str) -> List[float]:
        """
        Cohere Embed v4 이미지+텍스트 임베딩
        """
        if image_base64.startswith('/9j/'):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/png'
        
        image_uri = f"data:{mime_type};base64,{image_base64}"
        
        body = {
            "input_type": "search_document",
            "inputs": [
                {
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": image_uri}
                    ]
                }
            ],
            "embedding_types": ["float"]
        }
        
        return await self._invoke_cohere_v4_embedding(body)
    
    # ========== 내부 메서드 ==========
    
    async def _invoke_embedding(self, model_id: str, body: dict) -> List[float]:
        """임베딩 모델 호출 (Titan 계열)"""
        try:
            logger.debug(f"Invoking embedding model: {model_id}")
            
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            
            # 모델별 응답 파싱
            if "titan" in model_id:
                return response_body.get('embedding', [])
            elif "cohere" in model_id:
                embeddings = response_body.get('embeddings', [])
                return embeddings[0] if embeddings else []
            else:
                raise EmbeddingError(f"Unknown embedding model: {model_id}")
                
        except Exception as e:
            logger.error(f"Embedding generation failed for {model_id}: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
    
    async def _invoke_cohere_v4_embedding(self, body: dict) -> List[float]:
        """Cohere v4 임베딩 호출 (멀티모달용)"""
        try:
            model_id = settings.embedding_models['cohere_v4']
            logger.debug(f"Invoking Cohere v4 embedding: {model_id}")
            
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            
            # Cohere v4 응답 파싱
            embeddings = response_body.get('embeddings', {})
            
            # float 타입 임베딩 추출
            if 'float' in embeddings:
                return embeddings['float'][0] if embeddings['float'] else []
            
            # 기본 embeddings 배열
            if isinstance(embeddings, list) and embeddings:
                return embeddings[0]
            
            raise EmbeddingError("No embeddings in Cohere v4 response")
                
        except Exception as e:
            logger.error(f"Cohere v4 embedding failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate Cohere v4 embedding: {str(e)}")
