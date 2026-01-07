import logging
import json
from typing import Dict, Any, List
from app.orchestrator.context import ExecutionContext
from app.core.schemas import MetricScore, ExampleInput, PromptType
from app.core.config import settings

logger = logging.getLogger(__name__)

class RelevanceStage:
    """정확도 계산 단계 - AI 기반 조건 준수 평가"""
    
    def __init__(self, context: ExecutionContext):
        self.context = context
    
    async def execute(
        self, 
        prompt: str,
        example_inputs: List[ExampleInput],
        execution_results: Dict[str, Any],
        prompt_type: PromptType
    ) -> MetricScore:
        """
        정확도 점수 계산
        1. 입력 프롬프트에서 명시적 조건과 방향성 추출
        2. AI 판단을 통한 조건 준수 여부 평가
        3. 100점 만점 점수 계산
        """
        logger.info("Calculating accuracy score using AI-based evaluation")
        
        try:
            judge = self.context.get_judge()
            executions = execution_results['executions']
            
            all_accuracy_scores = []
            details = {'per_input_scores': [], 'extracted_conditions': []}
            
            for i, example_input in enumerate(example_inputs):
                logger.info(f"Processing input {i+1}/{len(example_inputs)}")
                
                # 1. 입력 프롬프트에서 조건과 방향성 추출
                conditions = await self._extract_conditions(judge, prompt, example_input.content)
                details['extracted_conditions'].append({
                    'input_index': i,
                    'conditions': conditions
                })
                
                if not conditions.get('explicit_conditions') and not conditions.get('direction'):
                    logger.warning(f"No conditions extracted for input {i}")
                    all_accuracy_scores.append(50.0)
                    continue
                
                # 해당 입력의 출력들 찾기
                exec_data = next((e for e in executions if e['input_index'] == i), None)
                if not exec_data:
                    logger.warning(f"No execution data for input {i}")
                    all_accuracy_scores.append(0.0)
                    continue
                
                # 2. 각 출력에 대해 조건 준수 평가
                output_scores = []
                evaluation_details = []
                
                for j, output in enumerate(exec_data['outputs']):
                    if not output or not output.strip():
                        output_scores.append(0.0)
                        continue
                    
                    # 이미지 타입인 경우 VLM 평가
                    if prompt_type == PromptType.TYPE_B_IMAGE:
                        evaluation = await self._evaluate_image_compliance(
                            judge, conditions, output, prompt
                        )
                    else:
                        # 텍스트 평가
                        evaluation = await self._evaluate_compliance(
                            judge, conditions, output, example_input.input_type, prompt_type
                        )
                    
                    score = self._calculate_compliance_score(evaluation)
                    output_scores.append(score)
                    evaluation_details.append({
                        'output_index': j,
                        'evaluation': evaluation,
                        'score': score
                    })
                
                # 해당 입력의 평균 점수
                input_score = sum(output_scores) / len(output_scores) if output_scores else 0.0
                all_accuracy_scores.append(input_score)
                
                details['per_input_scores'].append({
                    'input_index': i,
                    'score': input_score,
                    'output_count': len(output_scores),
                    'evaluation_details': evaluation_details
                })
            
            # 전체 평균 점수 (100점 만점)
            final_score = sum(all_accuracy_scores) / len(all_accuracy_scores) if all_accuracy_scores else 0.0
            
            details['final_score'] = final_score
            details['note'] = 'AI-based accuracy evaluation, score out of 100'
            
            logger.info(f"Accuracy score: {final_score:.3f}")
            return MetricScore(score=final_score, details=details)
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {str(e)}")
            return MetricScore(score=0.0, details={'error': str(e)})
    
    async def _extract_conditions(self, judge, prompt: str, input_content: str) -> Dict[str, Any]:
        """입력 프롬프트에서 명시적 조건과 방향성 추출"""
        
        extraction_prompt = f"""
다음 프롬프트를 분석하여 명시적 조건과 방향성을 추출해주세요.

프롬프트: {prompt}
입력 내용: {input_content[:500] if len(input_content) > 500 else input_content}

다음 형식으로 JSON 응답해주세요:
{{
    "explicit_conditions": [
        "조건1: 구체적인 요구사항",
        "조건2: 형식이나 길이 제한",
        "조건3: 포함해야 할 내용"
    ],
    "direction": "프롬프트가 지시하는 핵심 방향성과 목적"
}}

명시적 조건은 구체적으로 언급된 요구사항만 포함하고, 방향성은 전체적인 의도를 요약해주세요.
"""
        
        try:
            result = await judge.evaluate(extraction_prompt, "condition_extraction")
            if result.startswith('{') and result.endswith('}'):
                return json.loads(result)
            else:
                return {
                    "explicit_conditions": ["조건 추출 실패"],
                    "direction": result[:200]
                }
        except Exception as e:
            logger.error(f"Condition extraction failed: {str(e)}")
            return {
                "explicit_conditions": [],
                "direction": "방향성 추출 실패"
            }
    
    async def _evaluate_compliance(
        self, 
        judge, 
        conditions: Dict[str, Any], 
        output: str, 
        input_type: str,
        prompt_type: PromptType
    ) -> Dict[str, str]:
        """텍스트 출력에 대한 조건 준수 평가"""
        
        evaluation_prompt = f"""
다음 조건들이 출력에서 얼마나 잘 지켜졌는지 평가해주세요.

명시적 조건들:
{chr(10).join(f"- {cond}" for cond in conditions.get('explicit_conditions', []))}

방향성/핵심 과정:
{conditions.get('direction', '없음')}

출력 내용:
{output}

각 조건에 대해 다음 중 하나로 평가해주세요:
- "지킴": 조건을 명확히 준수함
- "안지킴": 조건을 명확히 위반함  
- "애매함": 판단하기 어렵거나 부분적으로만 준수

다음 JSON 형식으로 응답해주세요:
{{
    "explicit_conditions_compliance": [
        {{"condition": "조건1", "status": "지킴|안지킴|애매함", "reason": "판단 근거"}},
        {{"condition": "조건2", "status": "지킴|안지킴|애매함", "reason": "판단 근거"}}
    ],
    "direction_compliance": {{"status": "지킴|안지킴|애매함", "reason": "방향성 준수 여부와 근거"}},
    "overall_assessment": "전체적인 평가 요약"
}}
"""
        
        try:
            result = await judge.evaluate(evaluation_prompt, "compliance_evaluation")
            if result.startswith('{') and result.endswith('}'):
                return json.loads(result)
            else:
                return {
                    "explicit_conditions_compliance": [],
                    "direction_compliance": {"status": "애매함", "reason": "평가 실패"},
                    "overall_assessment": result[:200]
                }
        except Exception as e:
            logger.error(f"Compliance evaluation failed: {str(e)}")
            return {
                "explicit_conditions_compliance": [],
                "direction_compliance": {"status": "애매함", "reason": f"평가 오류: {str(e)}"},
                "overall_assessment": "평가 실패"
            }
    
    async def _evaluate_image_compliance(
        self, 
        judge, 
        conditions: Dict[str, Any], 
        image_base64: str,
        original_prompt: str
    ) -> Dict[str, str]:
        """
        VLM을 사용한 이미지 출력 조건 준수 평가
        Claude 3 Haiku Vision 사용
        """
        try:
            # 이미지가 유효한 base64인지 확인
            if not self._is_base64_image(image_base64):
                return {
                    "explicit_conditions_compliance": [],
                    "direction_compliance": {"status": "안지킴", "reason": "유효한 이미지가 아님"},
                    "overall_assessment": "이미지 생성 실패"
                }
            
            # VLM 평가 프롬프트
            evaluation_prompt = f"""이 이미지가 다음 조건들을 얼마나 잘 만족하는지 평가해주세요.

원본 프롬프트: {original_prompt}

명시적 조건들:
{chr(10).join(f"- {cond}" for cond in conditions.get('explicit_conditions', []))}

방향성/핵심 목적:
{conditions.get('direction', '없음')}

각 조건에 대해 다음 중 하나로 평가해주세요:
- "지킴": 이미지가 조건을 명확히 만족함
- "안지킴": 이미지가 조건을 명확히 위반함  
- "애매함": 판단하기 어렵거나 부분적으로만 만족

다음 JSON 형식으로 응답해주세요:
{{
    "explicit_conditions_compliance": [
        {{"condition": "조건1", "status": "지킴|안지킴|애매함", "reason": "판단 근거"}},
        {{"condition": "조건2", "status": "지킴|안지킴|애매함", "reason": "판단 근거"}}
    ],
    "direction_compliance": {{"status": "지킴|안지킴|애매함", "reason": "방향성 준수 여부와 근거"}},
    "overall_assessment": "전체적인 이미지 평가 요약"
}}
"""
            
            # VLM 호출 (이미지 포함)
            result = await judge.evaluate_image(evaluation_prompt, image_base64)
            
            if result.startswith('{') and result.endswith('}'):
                return json.loads(result)
            else:
                # JSON 추출 시도
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                
                return {
                    "explicit_conditions_compliance": [],
                    "direction_compliance": {"status": "애매함", "reason": "VLM 평가 파싱 실패"},
                    "overall_assessment": result[:200]
                }
                
        except Exception as e:
            logger.error(f"Image compliance evaluation failed: {str(e)}")
            return {
                "explicit_conditions_compliance": [],
                "direction_compliance": {"status": "애매함", "reason": f"VLM 평가 오류: {str(e)}"},
                "overall_assessment": "이미지 평가 실패"
            }
    
    def _is_base64_image(self, content: str) -> bool:
        """base64 이미지인지 확인"""
        if not content:
            return False
        
        content = content.strip()
        
        if content.startswith('iVBORw0KGgo'):  # PNG
            return True
        if content.startswith('/9j/'):  # JPEG
            return True
        
        if len(content) > 1000:
            try:
                import base64
                base64.b64decode(content[:100])
                return True
            except:
                pass
        
        return False
    
    def _calculate_compliance_score(self, evaluation: Dict[str, Any]) -> float:
        """평가 결과를 100점 만점 점수로 변환"""
        
        total_score = 0.0
        total_weight = 0.0
        
        # 명시적 조건 점수 (70% 가중치)
        explicit_conditions = evaluation.get('explicit_conditions_compliance', [])
        if explicit_conditions:
            condition_scores = []
            for cond_eval in explicit_conditions:
                status = cond_eval.get('status', '애매함')
                if status == '지킴':
                    condition_scores.append(100.0)
                elif status == '안지킴':
                    condition_scores.append(0.0)
                else:  # 애매함
                    condition_scores.append(50.0)
            
            if condition_scores:
                avg_condition_score = sum(condition_scores) / len(condition_scores)
                total_score += avg_condition_score * 0.7
                total_weight += 0.7
        
        # 방향성 점수 (30% 가중치)
        direction_compliance = evaluation.get('direction_compliance', {})
        direction_status = direction_compliance.get('status', '애매함')
        
        if direction_status == '지킴':
            direction_score = 100.0
        elif direction_status == '안지킴':
            direction_score = 0.0
        else:  # 애매함
            direction_score = 50.0
        
        total_score += direction_score * 0.3
        total_weight += 0.3
        
        # 가중 평균 계산
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 50.0
