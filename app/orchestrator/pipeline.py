import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from app.core.schemas import JobCreateRequest, EvaluationResult, MetricScore, PromptType
from app.orchestrator.context import ExecutionContext
from app.orchestrator.stages.run_stage import RunStage
from app.orchestrator.stages.token_stage import TokenStage
from app.orchestrator.stages.density_stage import DensityStage
from app.orchestrator.stages.embed_stage import EmbedStage
from app.orchestrator.stages.consistency_stage import ConsistencyStage
from app.orchestrator.stages.relevance_stage import RelevanceStage
from app.orchestrator.stages.variance_stage import VarianceStage
from app.orchestrator.stages.judge_stage import JudgeStage
from app.orchestrator.stages.aggregate_stage import AggregateStage
from app.core.config import settings

logger = logging.getLogger(__name__)

class Orchestrator:
    """전체 파이프라인 오케스트레이터 - 유일한 진실의 소스"""
    
    def __init__(self, context: ExecutionContext):
        self.context = context
        self.stages = {
            'run': RunStage(context),
            'token': TokenStage(context),
            'density': DensityStage(context),
            'embed': EmbedStage(context),
            'consistency': ConsistencyStage(context),
            'relevance': RelevanceStage(context),
            'variance': VarianceStage(context),
            'judge': JudgeStage(context),
            'aggregate': AggregateStage(context)
        }
    
    async def run(self, job_request: JobCreateRequest) -> EvaluationResult:
        """전체 파이프라인 실행 - 병렬 최적화"""
        logger.info(f"Starting parallel pipeline for prompt type: {job_request.prompt_type}")
        
        try:
            # Phase 1: 병렬 실행 - 모든 기본 작업을 동시에 처리
            logger.info("Phase 1: Starting parallel execution of all basic tasks")
            
            parallel_tasks = [
                # 1. 토큰 계산 (로컬, 즉시 완료)
                self._calculate_tokens_parallel(job_request.prompt),
                
                # 2-4. 메인 AI 실행 (각 입력별 5회)
                self._run_main_ai_for_input(job_request, 0),  # 입력 1
                self._run_main_ai_for_input(job_request, 1),  # 입력 2  
                self._run_main_ai_for_input(job_request, 2),  # 입력 3
                
                # 5-6. 다른 버전 AI들 (variance용)
                self._run_variance_models(job_request),
                
                # 7. 조건 추출 (정확도 지표용)
                self._extract_conditions_parallel(job_request)
            ]
            
            # 모든 작업을 병렬로 실행
            results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
            
            # 결과 파싱
            token_score = results[0] if not isinstance(results[0], Exception) else None
            main_outputs = {
                0: results[1] if not isinstance(results[1], Exception) else [],
                1: results[2] if not isinstance(results[2], Exception) else [],
                2: results[3] if not isinstance(results[3], Exception) else []
            }
            variance_outputs = results[4] if not isinstance(results[4], Exception) else {}
            condition_extractions = results[5] if not isinstance(results[5], Exception) else {}
            
            # execution_results 구성
            execution_results = self._build_execution_results(main_outputs, job_request)
            self._last_execution_results = execution_results
            
            logger.info("Phase 1 completed: All parallel tasks finished")
            
            # Phase 1.5: 메인 출력들로 추가 병렬 처리
            logger.info("Phase 1.5: Processing main outputs in parallel")
            
            # 모든 메인 출력 수집 (입력1,2,3의 각 5개 출력 = 총 15개)
            all_main_outputs = []
            for input_index, outputs in main_outputs.items():
                for output_index, output in enumerate(outputs):
                    if output.strip():  # 빈 출력 제외
                        all_main_outputs.append({
                            'input_index': input_index,
                            'output_index': output_index,
                            'output': output
                        })
            
            # 병렬 작업 구성
            output_parallel_tasks = []
            
            # 1. Cohere 임베딩 (모든 출력)
            for output_data in all_main_outputs:
                task = self._embed_with_cohere(output_data)
                output_parallel_tasks.append(('cohere', output_data['input_index'], output_data['output_index'], task))
            
            # 2. Nova/Titan 임베딩 (모든 출력)
            for output_data in all_main_outputs:
                task = self._embed_with_nova_titan(output_data, job_request.prompt_type)
                output_parallel_tasks.append(('nova_titan', output_data['input_index'], output_data['output_index'], task))
            
            # 3. Claim 추출 (TYPE_A인 경우만, 모든 출력)
            if job_request.prompt_type == PromptType.TYPE_A:
                for output_data in all_main_outputs:
                    task = self._extract_claims_from_output(output_data)
                    output_parallel_tasks.append(('claims', output_data['input_index'], output_data['output_index'], task))
            
            # 모든 출력 처리 작업을 병렬 실행
            logger.info(f"Starting {len(output_parallel_tasks)} parallel output processing tasks")
            output_results = await asyncio.gather(*[task for _, _, _, task in output_parallel_tasks], return_exceptions=True)
            
            # 결과 정리
            cohere_embeddings = {}
            nova_titan_embeddings = {}
            output_claims = {}
            
            for i, (task_type, input_idx, output_idx, _) in enumerate(output_parallel_tasks):
                result = output_results[i]
                if isinstance(result, Exception):
                    logger.error(f"Output processing failed for {task_type} input{input_idx} output{output_idx}: {str(result)}")
                    continue
                
                if task_type == 'cohere':
                    if input_idx not in cohere_embeddings:
                        cohere_embeddings[input_idx] = {}
                    cohere_embeddings[input_idx][output_idx] = result
                elif task_type == 'nova_titan':
                    if input_idx not in nova_titan_embeddings:
                        nova_titan_embeddings[input_idx] = {}
                    nova_titan_embeddings[input_idx][output_idx] = result
                elif task_type == 'claims':
                    if input_idx not in output_claims:
                        output_claims[input_idx] = {}
                    output_claims[input_idx][output_idx] = result
            
            logger.info("Phase 1.5 completed: All output processing finished")
            
            # Phase 2: 모든 지표 계산을 병렬로 처리
            logger.info("Phase 2: Starting parallel metric calculations")
            
            # Phase 2 병렬 작업 구성
            phase2_tasks = []
            
            # 1. 정보 밀도 계산 (TYPE_A, TYPE_B_TEXT만)
            if job_request.prompt_type in [PromptType.TYPE_A, PromptType.TYPE_B_TEXT]:
                task = self.stages['density'].execute(execution_results)
                phase2_tasks.append(('density', task))
            else:
                phase2_tasks.append(('density', self._create_none_task()))
            
            # 2. Variance 모델 출력들에 대한 nova 임베딩 (각 입력별로 병렬)
            variance_embedding_tasks = []
            for input_index, model_outputs in variance_outputs.items():
                for model_name, output in model_outputs.items():
                    if output.strip():  # 빈 출력 제외
                        task = self._embed_variance_output_with_nova(
                            output, input_index, model_name, job_request.prompt_type
                        )
                        variance_embedding_tasks.append((input_index, model_name, task))
            
            # Variance 임베딩들을 하나의 큰 태스크로 묶기
            if variance_embedding_tasks:
                combined_variance_task = self._process_variance_embeddings_parallel(variance_embedding_tasks)
                phase2_tasks.append(('variance_embeddings', combined_variance_task))
            else:
                phase2_tasks.append(('variance_embeddings', self._create_none_task()))
            
            # 3. 일관성 계산 (TYPE_A, TYPE_B_IMAGE, 임베딩 사용)
            if job_request.prompt_type in [PromptType.TYPE_A, PromptType.TYPE_B_IMAGE]:
                task = self._calculate_consistency_score(cohere_embeddings, nova_titan_embeddings)
                phase2_tasks.append(('consistency', task))
            else:
                phase2_tasks.append(('consistency', self._create_none_task()))
            
            # 4. 관련성 계산 (모든 타입, 조건은 이미 추출됨)
            task = self._calculate_relevance_score(
                condition_extractions, execution_results, job_request.prompt_type
            )
            phase2_tasks.append(('relevance', task))
            
            # 5. 환각 탐지 점수 계산 (TYPE_A만, claim은 이미 추출됨)
            if job_request.prompt_type == PromptType.TYPE_A and output_claims:
                task = self._calculate_hallucination_score_from_claims(output_claims, execution_results)
                phase2_tasks.append(('hallucination', task))
            else:
                phase2_tasks.append(('hallucination', self._create_none_task()))
            
            # Phase 2 모든 작업을 병렬 실행
            logger.info(f"Starting {len(phase2_tasks)} parallel Phase 2 tasks")
            phase2_results = await asyncio.gather(*[task for _, task in phase2_tasks], return_exceptions=True)
            
            # 결과 파싱
            density_score = None
            variance_embeddings = {}
            consistency_score = None
            relevance_score = None
            hallucination_score = None
            
            for i, (task_type, _) in enumerate(phase2_tasks):
                result = phase2_results[i]
                if isinstance(result, Exception):
                    logger.error(f"Phase 2 task {task_type} failed: {str(result)}")
                    continue
                
                if task_type == 'density':
                    density_score = result
                elif task_type == 'variance_embeddings':
                    variance_embeddings = result if result else {}
                elif task_type == 'consistency':
                    consistency_score = result
                elif task_type == 'relevance':
                    relevance_score = result
                elif task_type == 'hallucination':
                    hallucination_score = result
            
            logger.info("Phase 2 completed: All parallel metric calculations finished")
            
            # Phase 3: MCP 기반 claim 검증 및 최종 일관성 계산 (병렬)
            logger.info("Phase 3: Starting parallel claim verification and final consistency calculations")
            
            # Phase 3 병렬 작업 구성
            phase3_tasks = []
            
            # 0. 응답의 일관성 지표 계산 (이미 Phase 2에서 완료됨)
            # consistency_score는 이미 계산됨
            
            # 1. 버전별 일관성 지표 계산 (variance 임베딩 기반)
            if variance_embeddings:
                task = self._calculate_version_consistency_score(variance_embeddings)
                phase3_tasks.append(('version_consistency', task))
            else:
                phase3_tasks.append(('version_consistency', self._create_none_task()))
            
            # 2-4. 각 입력별 claim들에 대한 MCP 검증 (병렬)
            claim_verification_tasks = []
            if job_request.prompt_type == PromptType.TYPE_A and output_claims:
                for input_index, input_claims in output_claims.items():
                    for output_index, claims in input_claims.items():
                        if claims:  # claim이 있는 경우만
                            for claim_index, claim in enumerate(claims):
                                task = self._verify_claim_with_mcp(
                                    claim, input_index, output_index, claim_index
                                )
                                claim_verification_tasks.append((
                                    input_index, output_index, claim_index, claim, task
                                ))
            
            # Claim 검증들을 하나의 큰 태스크로 묶기
            if claim_verification_tasks:
                combined_claim_task = self._process_claim_verifications_parallel(claim_verification_tasks)
                phase3_tasks.append(('claim_verifications', combined_claim_task))
            else:
                phase3_tasks.append(('claim_verifications', self._create_none_task()))
            
            # Phase 3 모든 작업을 병렬 실행
            logger.info(f"Starting {len(phase3_tasks)} parallel Phase 3 tasks")
            phase3_results = await asyncio.gather(*[task for _, task in phase3_tasks], return_exceptions=True)
            
            # 결과 파싱
            version_consistency_score = None
            claim_verification_results = {}
            
            for i, (task_type, _) in enumerate(phase3_tasks):
                result = phase3_results[i]
                if isinstance(result, Exception):
                    logger.error(f"Phase 3 task {task_type} failed: {str(result)}")
                    continue
                
                if task_type == 'version_consistency':
                    version_consistency_score = result
                elif task_type == 'claim_verifications':
                    claim_verification_results = result if result else {}
            
            logger.info("Phase 3 completed: All parallel claim verifications finished")
            
            # Phase 4: 최종 환각 탐지 점수 계산
            logger.info("Phase 4: Calculating final hallucination score")
            
            final_hallucination_score = None
            if job_request.prompt_type == PromptType.TYPE_A and claim_verification_results:
                final_hallucination_score = await self._calculate_final_hallucination_score(
                    claim_verification_results, execution_results
                )
            else:
                final_hallucination_score = hallucination_score  # Phase 2에서 계산된 mock 점수 사용
            
            logger.info("Phase 4 completed: Final hallucination score calculated")
            
            # 모델 편차 점수 계산 (variance_outputs + variance_embeddings 활용)
            variance_score = await self._calculate_variance_score_with_embeddings(
                variance_outputs, variance_embeddings, job_request.prompt_type
            )
            
            # 최종 집계 (Phase 3, 4 결과 반영)
            final_result = await self.stages['aggregate'].execute(
                job_request.prompt_type,
                {
                    'token_usage': token_score,
                    'information_density': density_score,
                    'consistency': consistency_score,
                    'relevance': relevance_score,
                    'hallucination': final_hallucination_score,  # Phase 4에서 계산된 최종 점수
                    'model_variance': variance_score,
                    'version_consistency': version_consistency_score  # Phase 3에서 계산된 버전별 일관성
                }
            )
            
            # 실제 AI 출력 결과 포함
            final_result.execution_results = execution_results
            
            logger.info("Parallel pipeline completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Parallel pipeline execution failed: {str(e)}")
            raise
    
    async def _calculate_tokens_parallel(self, prompt: str) -> 'TokenMetricScore':
        """토큰 계산 (병렬용)"""
        return await self.stages['token'].execute(prompt, {})
    
    async def _run_main_ai_for_input(self, job_request: JobCreateRequest, input_index: int) -> List[str]:
        """특정 입력에 대해 메인 AI 5회 실행"""
        if input_index >= len(job_request.example_inputs):
            return []
        
        example_input = job_request.example_inputs[input_index]
        runner = self.context.get_runner()
        
        # 프롬프트 채우기
        filled_prompt = self._fill_prompt(job_request.prompt, example_input.content)
        
        # 5회 병렬 실행
        tasks = []
        for _ in range(job_request.repeat_count):
            task = runner.invoke(
                model=job_request.recommended_model or settings.default_models[job_request.prompt_type.value],
                prompt=filled_prompt,
                input_type=example_input.input_type
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공한 결과만 추출
        outputs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"AI execution failed for input {input_index}: {str(result)}")
                outputs.append("")
            else:
                outputs.append(result.get('output', ''))
        
        logger.info(f"Main AI completed for input {input_index}: {len(outputs)} outputs")
        return outputs
    
    async def _run_variance_models(self, job_request: JobCreateRequest) -> Dict[str, Any]:
        """다른 버전 AI들 실행 (variance 계산용)"""
        try:
            # 모델 목록 가져오기
            comparison_models = {
                PromptType.TYPE_A: [
                    "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "anthropic.claude-3-sonnet-20240229-v1:0",
                    "anthropic.claude-3-haiku-20240307-v1:0"
                ],
                PromptType.TYPE_B_TEXT: [
                    "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "anthropic.claude-3-sonnet-20240229-v1:0", 
                    "anthropic.claude-3-haiku-20240307-v1:0"
                ],
                PromptType.TYPE_B_IMAGE: [
                    "amazon.nova-canvas-v1:0",
                    "amazon.titan-image-generator-v1",
                    "amazon.titan-image-generator-v2:0"
                ]
            }
            
            models = comparison_models.get(job_request.prompt_type, [])
            if len(models) < 2:
                return {}
            
            runner = self.context.get_runner()
            variance_results = {}
            
            # 각 입력별로 모든 모델 실행
            for i, example_input in enumerate(job_request.example_inputs):
                filled_prompt = self._fill_prompt(job_request.prompt, example_input.content)
                
                # 모든 모델을 병렬로 실행
                model_tasks = []
                for model in models:
                    task = runner.invoke(
                        model=model,
                        prompt=filled_prompt,
                        input_type=example_input.input_type
                    )
                    model_tasks.append((model, task))
                
                # 병렬 실행
                model_results = await asyncio.gather(*[task for _, task in model_tasks], return_exceptions=True)
                
                # 결과 저장
                input_results = {}
                for j, (model, _) in enumerate(model_tasks):
                    result = model_results[j]
                    if isinstance(result, Exception):
                        logger.error(f"Variance model {model} failed for input {i}: {str(result)}")
                        input_results[model] = ""
                    else:
                        input_results[model] = result.get('output', '')
                
                variance_results[i] = input_results
            
            logger.info(f"Variance models completed: {len(variance_results)} inputs")
            return variance_results
            
        except Exception as e:
            logger.error(f"Variance models execution failed: {str(e)}")
            return {}
    
    async def _extract_conditions_parallel(self, job_request: JobCreateRequest) -> Dict[int, Dict[str, Any]]:
        """모든 입력에 대해 조건 추출 (병렬) - 정확도 지표용"""
        try:
            judge = self.context.get_judge()
            condition_results = {}
            
            # 각 입력별로 조건 추출
            tasks = []
            for i, example_input in enumerate(job_request.example_inputs):
                task = self._extract_conditions_for_input(judge, job_request.prompt, example_input.content, i)
                tasks.append((i, task))
            
            # 병렬 실행
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # 결과 저장
            for j, (input_index, _) in enumerate(tasks):
                result = results[j]
                if isinstance(result, Exception):
                    logger.error(f"Condition extraction failed for input {input_index}: {str(result)}")
                    condition_results[input_index] = {
                        "explicit_conditions": [],
                        "direction": "조건 추출 실패"
                    }
                else:
                    condition_results[input_index] = result
            
            logger.info(f"Condition extraction completed: {len(condition_results)} inputs")
            return condition_results
            
        except Exception as e:
            logger.error(f"Condition extraction failed: {str(e)}")
            return {}
    
    async def _extract_conditions_for_input(self, judge, prompt: str, input_content: str, input_index: int) -> Dict[str, Any]:
        """특정 입력에 대해 조건 추출"""
        try:
            extraction_prompt = f"""
다음 프롬프트를 분석하여 명시적 조건과 방향성을 추출해주세요.

프롬프트: {prompt}
입력 내용: {input_content}

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
            
            result = await judge.evaluate(extraction_prompt, "condition_extraction")
            
            # JSON 파싱 시도
            if result.startswith('{') and result.endswith('}'):
                import json
                return json.loads(result)
            else:
                # JSON이 아닌 경우 기본 구조 반환
                return {
                    "explicit_conditions": ["조건 추출 실패"],
                    "direction": result[:200]  # 처음 200자만
                }
                
        except Exception as e:
            logger.error(f"Condition extraction for input {input_index} failed: {str(e)}")
            return {
                "explicit_conditions": [],
                "direction": "조건 추출 실패"
            }
    
    async def _embed_with_cohere(self, output_data: Dict[str, Any]) -> List[float]:
        """Cohere로 출력 임베딩"""
        try:
            embedder = self.context.get_embedder()
            embedding = await embedder.embed_cohere(output_data['output'])
            return embedding
        except Exception as e:
            logger.error(f"Cohere embedding failed for input{output_data['input_index']} output{output_data['output_index']}: {str(e)}")
            return []
    
    async def _embed_with_nova_titan(self, output_data: Dict[str, Any], prompt_type: PromptType) -> List[float]:
        """Nova/Titan으로 출력 임베딩"""
        try:
            embedder = self.context.get_embedder()
            if prompt_type == PromptType.TYPE_B_IMAGE:
                embedding = await embedder.embed_multimodal(output_data['output'])
            else:
                embedding = await embedder.embed_text(output_data['output'])
            return embedding
        except Exception as e:
            logger.error(f"Nova/Titan embedding failed for input{output_data['input_index']} output{output_data['output_index']}: {str(e)}")
            return []
    
    async def _extract_claims_from_output(self, output_data: Dict[str, Any]) -> List[str]:
        """특정 출력에서 claim 추출"""
        try:
            judge = self.context.get_judge()
            
            prompt = f"""[작업]
아래 <분석대상> 안의 텍스트를 문장 단위로 분리한 뒤, 각 문장을 6가지 타입으로 분류하고 TYPE_1에 해당하는 문장만 추출하세요.

[6가지 타입 정의]
TYPE_1 (FACT_VERIFIABLE) - 추출 대상 ✓
: 외부 자료(뉴스, 위키, 공식문서 등)로 참/거짓 검증이 가능한 객관적 사실
: 날짜, 숫자, 인물, 사건 등 구체적 정보가 포함된 문장

TYPE_2 (FACT_UNVERIFIABLE) - 제외 ✗
: 사실처럼 보이지만 검증 불가능한 주장

TYPE_3 (OPINION) - 제외 ✗
: 주관적 판단, 평가, 의견 (좋다, 최고다, 아름답다 등)

TYPE_4 (PREDICTION) - 제외 ✗
: 미래 예측, 전망

TYPE_5 (INSTRUCTION) - 제외 ✗
: 방법, 절차, 사용법 설명

TYPE_6 (CREATIVE) - 제외 ✗
: 창작, 마케팅 문구, 가정/상상

[무조건 제외]
- 인사말, 맺음말
- 템플릿 변수 포함 문장
- 이 지시문 안의 예시들

<분석대상>
{output_data['output']}
</분석대상>

[출력 규칙]
- TYPE_1 문장만 한 줄에 하나씩 출력
- TYPE_1이 없으면 NONE 출력
- 분석대상 바깥의 텍스트는 절대 출력하지 마세요"""
            
            result = await judge.analyze_text(prompt)
            
            if result.strip().upper() == "NONE":
                return []
            
            # 결과를 줄별로 분리하여 반환
            claims = []
            for line in result.split('\n'):
                line = line.strip()
                if not line or line.upper() == "NONE":
                    continue
                if line.startswith("FACT_VERIFIABLE"):
                    continue
                if "{{" in line and "}}" in line:
                    continue
                # 인사말/불필요 문구 필터링
                skip_patterns = ["안녕하세요", "알겠습니다", "감사합니다", "네,", "좋습니다", 
                               "작성해 드리겠습니다", "소개합니다", "소개해드릴게요"]
                if any(pattern in line for pattern in skip_patterns):
                    continue
                claims.append(line)
            
            return claims
            
        except Exception as e:
            logger.error(f"Claim extraction failed for input{output_data['input_index']} output{output_data['output_index']}: {str(e)}")
            return []
    
    async def _calculate_consistency_score(self, cohere_embeddings: Dict, nova_titan_embeddings: Dict) -> 'MetricScore':
        """병렬로 생성된 임베딩으로 일관성 점수 계산"""
        try:
            from app.core.schemas import MetricScore
            import numpy as np
            
            all_consistency_scores = []
            details = {'per_input_scores': []}
            
            for input_index in cohere_embeddings.keys():
                cohere_embs = cohere_embeddings.get(input_index, {})
                nova_titan_embs = nova_titan_embeddings.get(input_index, {})
                
                # 유효한 임베딩만 필터링
                valid_cohere = [emb for emb in cohere_embs.values() if emb]
                valid_nova_titan = [emb for emb in nova_titan_embs.values() if emb]
                
                if len(valid_cohere) < 3 or len(valid_nova_titan) < 3:
                    logger.warning(f"Input {input_index}: Not enough valid embeddings")
                    all_consistency_scores.append(0.0)
                    continue
                
                # Centroid 기반 일관성 계산
                cohere_score = self._calculate_centroid_consistency(valid_cohere)
                nova_titan_score = self._calculate_centroid_consistency(valid_nova_titan)
                
                # 앙상블 평균
                ensemble_score = (cohere_score + nova_titan_score) / 2
                all_consistency_scores.append(ensemble_score)
                
                details['per_input_scores'].append({
                    'input_index': input_index,
                    'score': ensemble_score,
                    'cohere_score': cohere_score,
                    'nova_titan_score': nova_titan_score,
                    'valid_embeddings': len(valid_cohere)
                })
            
            final_score = sum(all_consistency_scores) / len(all_consistency_scores) if all_consistency_scores else 0.0
            details['final_score'] = final_score
            details['note'] = 'Parallel consistency calculation with pre-generated embeddings'
            
            return MetricScore(score=final_score, details=details)
            
        except Exception as e:
            logger.error(f"Consistency calculation failed: {str(e)}")
            return MetricScore(score=0.0, details={'error': str(e)})
    
    def _calculate_centroid_consistency(self, embeddings: List[List[float]]) -> float:
        """단일 모델의 centroid 기반 일관성 계산"""
        import numpy as np
        from app.core.config import settings
        
        embeddings = np.array(embeddings)  # (N, D)
        
        # 1. 중심 벡터 계산
        centroid = np.mean(embeddings, axis=0)
        
        # 2. 정규화
        centroid_norm = centroid / np.linalg.norm(centroid)
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 3. 중심으로부터의 cosine distance 계산
        distances = 1 - np.dot(emb_norm, centroid_norm)
        
        # 4. 평균 거리와 최대 거리
        mean_d = np.mean(distances)
        max_d = np.max(distances)
        
        # 5. 일관성 점수 (alpha로 최대 거리 패널티) - 음수 방지 및 100점 만점 변환
        consistency = 1 - (mean_d + settings.alpha * max_d)
        
        # 0-1 범위로 클리핑 후 100점 만점으로 변환
        consistency_score = max(0.0, min(1.0, consistency)) * 100
        
        return consistency_score
    
    async def _calculate_relevance_score(self, condition_extractions: Dict, execution_results: Dict[str, Any], prompt_type: PromptType) -> 'MetricScore':
        """추출된 조건들로 관련성 점수 계산"""
        try:
            from app.core.schemas import MetricScore
            import json
            
            judge = self.context.get_judge()
            executions = execution_results['executions']
            
            all_accuracy_scores = []
            details = {'per_input_scores': [], 'extracted_conditions': []}
            
            for exec_data in executions:
                input_index = exec_data['input_index']
                conditions = condition_extractions.get(input_index, {})
                
                details['extracted_conditions'].append({
                    'input_index': input_index,
                    'conditions': conditions
                })
                
                if not conditions.get('explicit_conditions') and not conditions.get('direction'):
                    logger.warning(f"No conditions extracted for input {input_index}")
                    all_accuracy_scores.append(50.0)
                    continue
                
                # 각 출력에 대해 조건 준수 평가
                output_scores = []
                evaluation_details = []
                
                for j, output in enumerate(exec_data['outputs']):
                    if not output.strip():
                        output_scores.append(0.0)
                        continue
                    
                    # AI 판단 요청
                    evaluation = await self._evaluate_compliance(
                        judge, conditions, output, exec_data.get('input_type', 'text'), prompt_type
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
                    'input_index': input_index,
                    'score': input_score,
                    'output_count': len(output_scores),
                    'evaluation_details': evaluation_details
                })
            
            final_score = sum(all_accuracy_scores) / len(all_accuracy_scores) if all_accuracy_scores else 0.0
            details['final_score'] = final_score
            details['note'] = 'Parallel relevance calculation with pre-extracted conditions'
            
            return MetricScore(score=final_score, details=details)
            
        except Exception as e:
            logger.error(f"Relevance calculation failed: {str(e)}")
            return MetricScore(score=0.0, details={'error': str(e)})
    
    async def _evaluate_compliance(self, judge, conditions: Dict[str, Any], output: str, input_type: str, prompt_type: PromptType) -> Dict[str, str]:
        """AI를 통한 조건 준수 평가"""
        
        # 이미지 출력인 경우 VLM 사용 고려
        model_note = ""
        if input_type == "image" or prompt_type == PromptType.TYPE_B_IMAGE:
            model_note = "(이미지 분석 가능한 모델 사용)"
        
        evaluation_prompt = f"""
다음 조건들이 출력에서 얼마나 잘 지켜졌는지 평가해주세요. {model_note}

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
                import json
                return json.loads(result)
            else:
                # JSON 파싱 실패시 기본 응답
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
            return 50.0  # 기본값
    
    async def _calculate_hallucination_score_from_claims(self, output_claims: Dict, execution_results: Dict[str, Any]) -> 'MetricScore':
        """추출된 claim들로 환각 탐지 점수 계산"""
        try:
            from app.core.schemas import MetricScore
            
            all_claim_scores = []
            details = {'per_input_analysis': [], 'total_claims': 0}
            
            for input_index, input_claims in output_claims.items():
                input_analysis = {
                    'input_index': input_index,
                    'outputs_analysis': []
                }
                
                for output_index, claims in input_claims.items():
                    if not claims:
                        # claim 없으면 중간 점수
                        all_claim_scores.append(60.0)
                        input_analysis['outputs_analysis'].append({
                            'output_index': output_index,
                            'claims': [],
                            'score': 60.0,
                            'reason': 'no_verifiable_claims'
                        })
                        continue
                    
                    # 각 claim에 대해 Mock 점수 계산 (실제로는 MCP 검증)
                    claim_results = []
                    for claim in claims:
                        # Mock 점수 (실제로는 MCP 기반 검증)
                        mock_score = 75.0  # 임시 점수
                        claim_results.append({
                            'claim': claim[:100] + '...' if len(claim) > 100 else claim,
                            'score': mock_score
                        })
                        all_claim_scores.append(mock_score)
                    
                    # 해당 출력의 평균 점수
                    output_score = sum(result['score'] for result in claim_results) / len(claim_results)
                    
                    input_analysis['outputs_analysis'].append({
                        'output_index': output_index,
                        'claims': claim_results,
                        'score': output_score,
                        'total_claims': len(claim_results)
                    })
                
                details['per_input_analysis'].append(input_analysis)
            
            final_score = sum(all_claim_scores) / len(all_claim_scores) if all_claim_scores else 60.0
            
            details.update({
                'final_score': final_score,
                'total_claims': len(all_claim_scores),
                'note': 'Parallel hallucination detection with pre-extracted claims'
            })
            
            return MetricScore(score=final_score, details=details)
            
        except Exception as e:
            logger.error(f"Hallucination score calculation failed: {str(e)}")
            return MetricScore(score=0.0, details={'error': str(e)})
    
    def _build_execution_results(self, main_outputs: Dict[int, List[str]], job_request: JobCreateRequest) -> Dict[str, Any]:
        """메인 출력들로 execution_results 구성"""
        executions = []
        
        for input_index in range(len(job_request.example_inputs)):
            example_input = job_request.example_inputs[input_index]
            outputs = main_outputs.get(input_index, [])
            
            # 프롬프트 채우기
            filled_prompt = self._fill_prompt(job_request.prompt, example_input.content)
            
            executions.append({
                'input_index': input_index,
                'input_content': example_input.content,
                'filled_prompt': filled_prompt,
                'outputs': outputs,
                'model': job_request.recommended_model or settings.default_models[job_request.prompt_type.value]
            })
        
        return {'executions': executions}
    
    async def _calculate_hallucination_score(self, claim_extractions: Dict[int, List[str]], execution_results: Dict[str, Any]) -> 'MetricScore':
        """추출된 claim들로 환각 탐지 점수 계산"""
        try:
            from app.core.schemas import MetricScore
            
            all_claim_scores = []
            details = {'per_input_analysis': [], 'total_claims': 0}
            
            for input_index, claims in claim_extractions.items():
                if not claims:
                    all_claim_scores.append(60.0)  # claim 없으면 중간 점수
                    continue
                
                # 각 claim에 대해 Mock 점수 계산 (실제로는 MCP 검증)
                for claim in claims:
                    # Mock 점수 (실제로는 MCP 기반 검증)
                    mock_score = 75.0  # 임시 점수
                    all_claim_scores.append(mock_score)
            
            final_score = sum(all_claim_scores) / len(all_claim_scores) if all_claim_scores else 60.0
            
            details.update({
                'final_score': final_score,
                'total_claims': len(all_claim_scores),
                'note': 'Parallel hallucination detection with pre-extracted claims'
            })
            
            return MetricScore(score=final_score, details=details)
            
        except Exception as e:
            logger.error(f"Hallucination score calculation failed: {str(e)}")
            return MetricScore(score=0.0, details={'error': str(e)})
    
    def _calculate_model_variance(self, embeddings: Dict[str, List[float]]) -> float:
        """모델 간 임베딩 유사도 기반 일관성 계산"""
        import numpy as np
        
        if len(embeddings) < 2:
            return 0.0
        
        # 모든 임베딩을 numpy 배열로 변환
        embedding_arrays = {}
        for model, embedding in embeddings.items():
            embedding_arrays[model] = np.array(embedding)
        
        # 모든 모델 쌍의 코사인 유사도 계산
        similarities = []
        models = list(embedding_arrays.keys())
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                emb1, emb2 = embedding_arrays[model1], embedding_arrays[model2]
                
                # 코사인 유사도 계산
                similarity = self._cosine_similarity(emb1, emb2)
                similarities.append(similarity)
        
        # 평균 유사도
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return max(0.0, min(1.0, avg_similarity))
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """코사인 유사도 계산"""
        import numpy as np
        
        # 정규화
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # 코사인 유사도
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # 0-1 범위로 정규화
        return (similarity + 1) / 2
    
    def _fill_prompt(self, prompt: str, input_content: str) -> str:
        """프롬프트의 {{변수명}} 플레이스홀더를 실제 입력으로 치환"""
        import json
        import re
        
        result = prompt
        has_placeholder = bool(re.search(r'\{\{.*?\}\}', prompt))
        
        # 1. input_content가 JSON인 경우 파싱해서 각 키별로 치환
        try:
            data = json.loads(input_content)
            if isinstance(data, dict):
                for key, value in data.items():
                    result = result.replace(f"{{{{{key}}}}}", str(value))
        except (json.JSONDecodeError, TypeError):
            pass
        
        # 2. 기본 플레이스홀더 치환
        result = result.replace("{{}}", input_content).replace("{{input}}", input_content)
        
        # 3. 플레이스홀더가 없었으면 맨 뒤에 입력 추가
        if not has_placeholder:
            result = f"{result}\n\n{input_content}"
        
        return result
    
    async def _create_none_task(self):
        """빈 태스크 생성 (조건부 실행용)"""
        return None
    
    async def _embed_variance_output_with_nova(self, output: str, input_index: int, model_name: str, prompt_type: PromptType) -> List[float]:
        """Variance 출력을 Nova/Titan으로 임베딩"""
        try:
            embedder = self.context.get_embedder()
            if prompt_type == PromptType.TYPE_B_IMAGE:
                embedding = await embedder.embed_multimodal(output)
            else:
                embedding = await embedder.embed_text(output)
            return embedding
        except Exception as e:
            logger.error(f"Variance embedding failed for {model_name} input{input_index}: {str(e)}")
            return []
    
    async def _process_variance_embeddings_parallel(self, variance_embedding_tasks) -> Dict[str, Any]:
        """Variance 임베딩들을 병렬로 처리"""
        try:
            # 모든 variance 임베딩 태스크를 병렬 실행
            results = await asyncio.gather(*[task for _, _, task in variance_embedding_tasks], return_exceptions=True)
            
            # 결과 정리
            variance_embeddings = {}
            for i, (input_index, model_name, _) in enumerate(variance_embedding_tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Variance embedding failed for {model_name} input{input_index}: {str(result)}")
                    continue
                
                if input_index not in variance_embeddings:
                    variance_embeddings[input_index] = {}
                variance_embeddings[input_index][model_name] = result
            
            logger.info(f"Variance embeddings completed: {len(variance_embeddings)} inputs processed")
            return variance_embeddings
            
        except Exception as e:
            logger.error(f"Variance embeddings processing failed: {str(e)}")
            return {}
    
    async def _calculate_variance_score_with_embeddings(self, variance_outputs: Dict[str, Any], variance_embeddings: Dict[str, Any], prompt_type: PromptType) -> 'MetricScore':
        """variance 출력들과 임베딩으로 모델 편차 점수 계산 (개선된 버전)"""
        try:
            from app.core.schemas import MetricScore
            
            if not variance_outputs and not variance_embeddings:
                return MetricScore(score=50.0, details={'error': 'no_variance_data'})
            
            all_variance_scores = []
            details = {'per_input_analysis': []}
            
            # 각 입력별로 모델 간 유사도 계산
            for input_index in variance_outputs.keys():
                input_analysis = {
                    'input_index': input_index,
                    'model_similarities': []
                }
                
                # 해당 입력의 임베딩들 가져오기
                input_embeddings = variance_embeddings.get(input_index, {})
                
                if len(input_embeddings) >= 2:
                    # 모델 간 유사도 계산
                    variance_score = self._calculate_model_variance(input_embeddings)
                    all_variance_scores.append(variance_score)
                    
                    input_analysis['variance_score'] = variance_score
                    input_analysis['models_compared'] = list(input_embeddings.keys())
                else:
                    # 임베딩이 부족한 경우 기본 점수
                    all_variance_scores.append(0.5)
                    input_analysis['variance_score'] = 0.5
                    input_analysis['note'] = 'insufficient_embeddings'
                
                details['per_input_analysis'].append(input_analysis)
            
            # 최종 점수 계산 (0-1 범위를 100점 만점으로 변환)
            final_score = (sum(all_variance_scores) / len(all_variance_scores) * 100) if all_variance_scores else 50.0
            
            details.update({
                'final_score': final_score,
                'processed_inputs': len(all_variance_scores),
                'note': 'Parallel model variance calculation with pre-generated embeddings'
            })
            
            return MetricScore(score=final_score, details=details)
            
        except Exception as e:
            logger.error(f"Variance score calculation failed: {str(e)}")
            return MetricScore(score=0.0, details={'error': str(e)})
    
    async def _calculate_version_consistency_score(self, variance_embeddings: Dict[str, Any]) -> 'MetricScore':
        """버전별 일관성 점수 계산 (variance 임베딩 기반)"""
        try:
            from app.core.schemas import MetricScore
            import numpy as np
            
            all_consistency_scores = []
            details = {'per_input_analysis': []}
            
            for input_index, model_embeddings in variance_embeddings.items():
                if len(model_embeddings) < 2:
                    continue
                
                # 해당 입력의 모든 모델 임베딩들 간 유사도 계산
                embeddings_list = list(model_embeddings.values())
                valid_embeddings = [emb for emb in embeddings_list if emb]
                
                if len(valid_embeddings) >= 2:
                    # Centroid 기반 일관성 계산
                    consistency_score = self._calculate_centroid_consistency(valid_embeddings)
                    all_consistency_scores.append(consistency_score)
                    
                    details['per_input_analysis'].append({
                        'input_index': input_index,
                        'consistency_score': consistency_score,
                        'models_compared': list(model_embeddings.keys()),
                        'valid_embeddings': len(valid_embeddings)
                    })
            
            final_score = sum(all_consistency_scores) / len(all_consistency_scores) if all_consistency_scores else 50.0
            
            details.update({
                'final_score': final_score,
                'processed_inputs': len(all_consistency_scores),
                'note': 'Version consistency calculation based on variance model embeddings'
            })
            
            return MetricScore(score=final_score, details=details)
            
        except Exception as e:
            logger.error(f"Version consistency calculation failed: {str(e)}")
            return MetricScore(score=0.0, details={'error': str(e)})
    
    async def _verify_claim_with_mcp(self, claim: str, input_index: int, output_index: int, claim_index: int) -> Dict[str, Any]:
        """MCP를 통한 개별 claim 검증"""
        try:
            # Mock MCP 검증 (실제로는 MCP 서버 호출)
            # 여기서는 임시로 mock 결과 반환
            import random
            
            # 실제 구현에서는 MCP 서버를 통해 claim을 검증
            # 예: web search, knowledge base 조회 등
            
            verification_score = random.uniform(0.6, 0.9)  # Mock 점수
            evidence_found = random.choice([True, False])
            
            return {
                'claim': claim[:100] + '...' if len(claim) > 100 else claim,
                'input_index': input_index,
                'output_index': output_index,
                'claim_index': claim_index,
                'verification_score': verification_score,
                'evidence_found': evidence_found,
                'evidence_sources': ['mock_source_1', 'mock_source_2'] if evidence_found else [],
                'verification_method': 'mcp_mock'
            }
            
        except Exception as e:
            logger.error(f"Claim verification failed for claim {claim_index}: {str(e)}")
            return {
                'claim': claim[:100] + '...' if len(claim) > 100 else claim,
                'input_index': input_index,
                'output_index': output_index,
                'claim_index': claim_index,
                'verification_score': 0.0,
                'evidence_found': False,
                'evidence_sources': [],
                'verification_method': 'error',
                'error': str(e)
            }
    
    async def _process_claim_verifications_parallel(self, claim_verification_tasks) -> Dict[str, Any]:
        """모든 claim 검증을 병렬로 처리"""
        try:
            # 모든 claim 검증 태스크를 병렬 실행
            results = await asyncio.gather(*[task for _, _, _, _, task in claim_verification_tasks], return_exceptions=True)
            
            # 결과 정리
            verification_results = {}
            total_verifications = 0
            successful_verifications = 0
            
            for i, (input_index, output_index, claim_index, claim, _) in enumerate(claim_verification_tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Claim verification failed for input{input_index} output{output_index} claim{claim_index}: {str(result)}")
                    continue
                
                # 결과 저장 구조: input_index -> output_index -> claim_index -> result
                if input_index not in verification_results:
                    verification_results[input_index] = {}
                if output_index not in verification_results[input_index]:
                    verification_results[input_index][output_index] = {}
                
                verification_results[input_index][output_index][claim_index] = result
                total_verifications += 1
                
                if result.get('evidence_found', False):
                    successful_verifications += 1
            
            logger.info(f"Claim verifications completed: {successful_verifications}/{total_verifications} claims verified")
            
            return {
                'verification_results': verification_results,
                'total_claims': total_verifications,
                'verified_claims': successful_verifications,
                'verification_rate': successful_verifications / total_verifications if total_verifications > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Claim verifications processing failed: {str(e)}")
            return {}
    
    async def _calculate_final_hallucination_score(self, claim_verification_results: Dict[str, Any], execution_results: Dict[str, Any]) -> 'MetricScore':
        """MCP 검증 결과를 바탕으로 최종 환각 탐지 점수 계산"""
        try:
            from app.core.schemas import MetricScore
            
            verification_data = claim_verification_results.get('verification_results', {})
            if not verification_data:
                return MetricScore(score=60.0, details={'error': 'no_verification_data'})
            
            all_scores = []
            details = {'per_input_analysis': []}
            
            for input_index, input_verifications in verification_data.items():
                input_analysis = {
                    'input_index': input_index,
                    'outputs_analysis': []
                }
                
                for output_index, output_verifications in input_verifications.items():
                    output_scores = []
                    claim_details = []
                    
                    for claim_index, verification in output_verifications.items():
                        score = verification.get('verification_score', 0.0) * 100  # 0-1을 0-100으로 변환
                        output_scores.append(score)
                        all_scores.append(score)
                        
                        claim_details.append({
                            'claim_index': claim_index,
                            'claim': verification.get('claim', ''),
                            'score': score,
                            'evidence_found': verification.get('evidence_found', False),
                            'sources': verification.get('evidence_sources', [])
                        })
                    
                    # 해당 출력의 평균 점수
                    output_score = sum(output_scores) / len(output_scores) if output_scores else 60.0
                    
                    input_analysis['outputs_analysis'].append({
                        'output_index': output_index,
                        'score': output_score,
                        'total_claims': len(claim_details),
                        'claim_details': claim_details
                    })
                
                details['per_input_analysis'].append(input_analysis)
            
            # 최종 점수 계산
            final_score = sum(all_scores) / len(all_scores) if all_scores else 60.0
            
            details.update({
                'final_score': final_score,
                'total_claims_verified': len(all_scores),
                'verification_rate': claim_verification_results.get('verification_rate', 0.0),
                'note': 'Final hallucination score based on MCP claim verification'
            })
            
            return MetricScore(score=final_score, details=details)
            
        except Exception as e:
            logger.error(f"Final hallucination score calculation failed: {str(e)}")
            return MetricScore(score=0.0, details={'error': str(e)})