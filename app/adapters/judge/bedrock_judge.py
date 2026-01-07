import json
import logging
import boto3
from app.adapters.judge.base import BaseJudge
from app.core.config import settings
from app.core.errors import ModelInvocationError

logger = logging.getLogger(__name__)

class BedrockJudge(BaseJudge):
    """AWS Bedrock 기반 환각 탐지 Judge"""
    
    def __init__(self):
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None
        )
        self.judge_model = settings.judge_model
    
    async def judge_factuality(self, question: str, answer: str) -> bool:
        """사실성 판별"""
        try:
            prompt = self._build_judge_prompt(question, answer)
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "temperature": 0.1,  # 낮은 온도로 일관된 판별
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.client.invoke_model(
                modelId=self.judge_model,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body.get('content', [])
            judgment = content[0].get('text', '').strip().upper() if content else ''
            
            # TRUE/FALSE 판별
            return judgment.startswith('TRUE')
            
        except Exception as e:
            logger.error(f"Judge invocation failed: {str(e)}")
            # 실패시 보수적으로 FALSE 반환 (환각으로 간주)
            return False
    
    async def evaluate(self, prompt: str, task_type: str = "general") -> str:
        """범용 AI 평가 메서드"""
        try:
            # 이미지 관련 작업인 경우 VLM 모델 사용 고려
            model_id = self.judge_model
            if "image" in task_type.lower():
                # 이미지 처리 가능한 모델로 변경 (설정에 따라)
                model_id = getattr(settings, 'vlm_judge_model', self.judge_model)
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,  # 더 긴 응답 허용
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body.get('content', [])
            result = content[0].get('text', '') if content else ''
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"AI evaluation failed: {str(e)}")
            return f"평가 실패: {str(e)}"
    
    async def analyze_text(self, prompt: str) -> str:
        """텍스트 분석용 범용 메서드 (환각 탐지용)"""
        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.1,  # 낮은 온도로 일관된 분석
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.client.invoke_model(
                modelId=self.judge_model,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body.get('content', [])
            result = content[0].get('text', '') if content else ''
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return f"분석 실패: {str(e)}"
    
    async def evaluate_image(self, prompt: str, image_base64: str) -> str:
        """
        VLM을 사용한 이미지 평가 메서드
        Claude 3 Haiku Vision 사용
        """
        try:
            # 이미지 타입 감지
            if image_base64.startswith('/9j/'):
                media_type = 'image/jpeg'
            else:
                media_type = 'image/png'
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            response = self.client.invoke_model(
                modelId=self.judge_model,  # Claude 3 Haiku (VLM 지원)
                body=json.dumps(body),
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body.get('content', [])
            result = content[0].get('text', '') if content else ''
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Image evaluation failed: {str(e)}")
            return f"이미지 평가 실패: {str(e)}"
    
    def _build_judge_prompt(self, question: str, answer: str) -> str:
        return f"""You are a factuality judge. Evaluate if the given answer contains factual information that can be verified or is reasonable based on the question.

Question: {question}

Answer: {answer}

Instructions:
- Respond with "TRUE" if the answer contains factual, verifiable, or reasonable information
- Respond with "FALSE" if the answer contains obvious hallucinations, false claims, or unverifiable statements
- Focus on factual accuracy, not completeness or style

Response (TRUE or FALSE only):"""