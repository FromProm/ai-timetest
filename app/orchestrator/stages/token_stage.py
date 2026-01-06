import logging
import tiktoken
from typing import Dict, Any
from app.orchestrator.context import ExecutionContext
from app.core.schemas import TokenMetricScore

logger = logging.getLogger(__name__)

class TokenStage:
    """토큰 사용량 계산 단계"""
    
    def __init__(self, context: ExecutionContext):
        self.context = context
        # GPT 계열 토크나이저 (근사치로 사용)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def execute(self, prompt: str, execution_results: Dict[str, Any]) -> TokenMetricScore:
        """
        토큰 사용량 계산
        - 고정 프롬프트 부분의 토큰 수 계산 ({{}} 제외)
        """
        logger.info("Calculating token usage")
        
        try:
            # 고정 프롬프트 토큰 수 계산 (플레이스홀더 제거)
            fixed_prompt = self._remove_placeholders(prompt)
            fixed_tokens = len(self.tokenizer.encode(fixed_prompt))
            
            details = {
                'fixed_prompt_tokens': fixed_tokens,
                'fixed_prompt_text': fixed_prompt
            }
            
            logger.info(f"Fixed prompt token count: {fixed_tokens}")
            return TokenMetricScore(score=float(fixed_tokens), details=details)
            
        except Exception as e:
            logger.error(f"Token calculation failed: {str(e)}")
            return TokenMetricScore(score=0.0, details={'error': str(e)})
    
    def _remove_placeholders(self, prompt: str) -> str:
        """플레이스홀더 제거"""
        # {{}} 형태의 플레이스홀더 제거
        import re
        return re.sub(r'\{\{[^}]*\}\}', '', prompt).strip()