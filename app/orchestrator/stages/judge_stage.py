import logging
import asyncio
import re
from typing import Dict, Any, List, Optional
from app.orchestrator.context import ExecutionContext
from app.core.schemas import MetricScore, ExampleInput, ClaimType, Verdict
from app.adapters.mcp.mcp_client import MCPClient

logger = logging.getLogger(__name__)

class JudgeStage:
    """환각 탐지 단계 - MCP 기반 사실 검증"""
    
    def __init__(self, context: ExecutionContext):
        self.context = context
        self.mcp_client = MCPClient()
        # Verdict별 점수 매핑 (환각 탐지 관점)
        self.verdict_scores = {
            Verdict.SUPPORTED: 1.0,      # 확인된 사실 → 환각 없음 (최고점)
            Verdict.INSUFFICIENT: 0.7,   # 검증 불가 → 환각 의심 (중간점)
            Verdict.REFUTED: 0.0         # 명확한 거짓 → 확실한 환각 (최저점)
        }
    
    async def execute(
        self, 
        example_inputs: List[ExampleInput], 
        execution_results: Dict[str, Any]
    ) -> MetricScore:
        """
        환각 탐지 점수 계산
        1. 출력에서 FACT_VERIFIABLE 문장 추출
        2. MCP로 외부 근거 수집
        3. 근거 기반 SUPPORTED/REFUTED/INSUFFICIENT 판정
        4. 점수 계산 (100점 만점)
        """
        logger.info("Running hallucination detection with MCP fact verification")
        
        try:
            # MCP 클라이언트 초기화
            await self.mcp_client.initialize()
            
            judge = self.context.get_judge()
            executions = execution_results['executions']
            
            all_claim_scores = []
            details = {'per_input_analysis': [], 'total_claims': 0, 'verdict_distribution': {}}
            
            for exec_data in executions:
                input_index = exec_data['input_index']
                input_content = exec_data['input_content']
                outputs = exec_data['outputs']
                
                logger.info(f"Analyzing outputs for input {input_index+1}")
                
                input_analysis = {
                    'input_index': input_index,
                    'input_content': input_content[:100] + '...' if len(input_content) > 100 else input_content,
                    'outputs_analysis': []
                }
                
                # 각 출력별로 분석
                for output_idx, output in enumerate(outputs):
                    if not output.strip():
                        input_analysis['outputs_analysis'].append({
                            'output_index': output_idx,
                            'output_preview': '',
                            'claims': [],
                            'score': 0.0,
                            'reason': 'empty_output'
                        })
                        all_claim_scores.append(0.0)
                        continue
                    
                    # 1단계: FACT_VERIFIABLE 문장 추출
                    verifiable_claims = await self._extract_verifiable_claims(output)
                    
                    if not verifiable_claims:
                        input_analysis['outputs_analysis'].append({
                            'output_index': output_idx,
                            'output_preview': output[:100] + '...' if len(output) > 100 else output,
                            'claims': [],
                            'score': 60.0,
                            'reason': 'no_verifiable_claims'
                        })
                        all_claim_scores.append(60.0)
                        continue
                    
                    # 2-3단계: 각 claim에 대해 MCP 검증 및 점수 계산
                    claim_results = []
                    for claim in verifiable_claims:
                        claim_score = await self._verify_claim_with_mcp(claim)
                        claim_score_100 = claim_score * 100
                        
                        claim_results.append({
                            'claim': claim[:100] + '...' if len(claim) > 100 else claim,
                            'score': claim_score_100
                        })
                        all_claim_scores.append(claim_score_100)
                    
                    output_score = sum(result['score'] for result in claim_results) / len(claim_results)
                    
                    input_analysis['outputs_analysis'].append({
                        'output_index': output_idx,
                        'output_preview': output[:100] + '...' if len(output) > 100 else output,
                        'claims': claim_results,
                        'score': output_score,
                        'total_claims': len(claim_results)
                    })
                
                details['per_input_analysis'].append(input_analysis)
            
            final_score = sum(all_claim_scores) / len(all_claim_scores) if all_claim_scores else 0.0
            
            score_distribution = {
                'perfect_scores': 0,
                'partial_scores': 0,
                'zero_scores': 0
            }
            
            for score in all_claim_scores:
                if score == 100.0:
                    score_distribution['perfect_scores'] += 1
                elif score == 0.0:
                    score_distribution['zero_scores'] += 1
                else:
                    score_distribution['partial_scores'] += 1
            
            details.update({
                'final_score': final_score,
                'total_claims': len(all_claim_scores),
                'score_distribution': score_distribution,
                'note': 'Hallucination detection using real MCP servers (Wikipedia, ArXiv, DuckDuckGo, Fetch)'
            })
            
            logger.info(f"Hallucination detection score: {final_score:.3f} (total claims: {len(all_claim_scores)})")
            return MetricScore(score=final_score, details=details)
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {str(e)}")
            return MetricScore(score=0.0, details={'error': str(e)})
        finally:
            # MCP 클라이언트 정리
            await self.mcp_client.close()

    async def _extract_verifiable_claims(self, output: str) -> List[str]:
        """출력에서 FACT_VERIFIABLE 타입의 문장들을 추출"""
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
{output}
</분석대상>

[출력 규칙]
- TYPE_1 문장만 한 줄에 하나씩 출력
- TYPE_1이 없으면 NONE 출력
- 분석대상 바깥의 텍스트는 절대 출력하지 마세요"""
            
            result = await judge.analyze_text(prompt)
            
            if result.strip().upper() == "NONE":
                return []
            
            claims = []
            for line in result.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.upper() == "NONE":
                    continue
                if line.startswith("FACT_VERIFIABLE"):
                    continue
                if "{{" in line and "}}" in line:
                    continue
                skip_patterns = ["안녕하세요", "알겠습니다", "감사합니다", "네,", "좋습니다", 
                               "작성해 드리겠습니다", "소개합니다", "소개해드릴게요"]
                if any(pattern in line for pattern in skip_patterns):
                    continue
                claims.append(line)
            
            return claims
            
        except Exception as e:
            logger.error(f"Failed to extract verifiable claims: {str(e)}")
            return []
    
    async def _verify_claim_with_mcp(self, claim: str) -> float:
        """실제 MCP로 claim 검증하여 점수 반환"""
        try:
            logger.info(f"Verifying claim: {claim[:100]}...")
            
            # 1. AI가 이 claim에 가장 적합한 MCP 선택
            selected_mcp = await self._select_best_mcp_for_claim(claim)
            logger.info(f"Selected MCP: {selected_mcp}")
            
            # 2. 선택된 MCP로 근거 수집 (실제 MCP 호출)
            evidence = await self._collect_evidence_from_mcp(claim, selected_mcp)
            
            if not evidence:
                logger.warning(f"No evidence found for claim: {claim[:50]}...")
                return 0.5  # 근거 없으면 중간 점수
            
            # 3. 근거 품질 평가
            quality_score = self._evaluate_evidence_quality(evidence)
            
            # 4. 근거 기반 점수 계산
            score = await self._judge_claim_with_evidence(claim, evidence, quality_score)
            
            logger.info(f"Final score: {score:.3f}/1.0 → {score*100:.1f}/100 (mcp: {selected_mcp})")
            return score
            
        except Exception as e:
            logger.error(f"Failed to verify claim with MCP: {str(e)}")
            return 0.0
    
    async def _select_best_mcp_for_claim(self, claim: str) -> str:
        """AI가 claim에 가장 적합한 MCP 선택"""
        try:
            judge = self.context.get_judge()
            
            prompt = f"""
다음 claim을 검증하기 위해 가장 적합한 MCP 하나를 선택해주세요:

Claim: {claim}

사용 가능한 MCP:
1. DUCKDUCKGO - 일반 웹 검색 (연도/숫자/사실 확인, 최신 정보)
2. WIKIPEDIA - 위키피디아 검색 (인물/역사/기본 사실, 높은 신뢰도)
3. ARXIV - 학술 논문 검색 (과학/의학/연구 분야)
4. WEB_SCRAPER - 특정 페이지 상세 분석 (공식 발표/뉴스 원문)

선택 기준:
- 연도/숫자/최신 정보 → DUCKDUCKGO
- 인물/역사/일반 상식 → WIKIPEDIA  
- 과학/의학/연구 → ARXIV
- 공식 발표/뉴스 원문 → WEB_SCRAPER

가장 적합한 MCP 하나만 반환하세요: DUCKDUCKGO, WIKIPEDIA, ARXIV, WEB_SCRAPER 중 하나
"""
            
            result = await judge.analyze_text(prompt)
            selected_mcp = result.strip().upper()
            
            # 유효한 MCP 타입인지 확인
            valid_types = ['DUCKDUCKGO', 'WIKIPEDIA', 'ARXIV', 'WEB_SCRAPER']
            if selected_mcp not in valid_types:
                # 부분 매칭 시도
                for valid_type in valid_types:
                    if valid_type in selected_mcp:
                        return valid_type
                return 'DUCKDUCKGO'  # 기본값
            
            return selected_mcp
            
        except Exception as e:
            logger.error(f"Failed to select MCP: {str(e)}")
            return 'DUCKDUCKGO'
    
    async def _collect_evidence_from_mcp(self, claim: str, mcp_type: str) -> List[Dict[str, Any]]:
        """실제 MCP 서버에서 근거 수집"""
        try:
            # 검색 쿼리 생성
            search_query = await self._generate_search_query(claim)
            logger.info(f"Search query: {search_query}")
            
            if mcp_type == 'DUCKDUCKGO':
                return await self._collect_from_duckduckgo(search_query)
            elif mcp_type == 'WIKIPEDIA':
                return await self._collect_from_wikipedia(search_query)
            elif mcp_type == 'ARXIV':
                return await self._collect_from_arxiv(search_query)
            elif mcp_type == 'WEB_SCRAPER':
                return await self._collect_from_web_scraper(search_query)
            else:
                return await self._collect_from_duckduckgo(search_query)
                
        except Exception as e:
            logger.error(f"Failed to collect evidence from {mcp_type}: {str(e)}")
            return []
    
    async def _collect_from_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """DuckDuckGo MCP로 웹 검색"""
        try:
            results = await self.mcp_client.search_web(query)
            logger.info(f"DuckDuckGo returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []
    
    async def _collect_from_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        """Wikipedia MCP로 검색"""
        try:
            # 먼저 검색
            search_results = await self.mcp_client.search_wikipedia(query)
            
            if not search_results:
                return []
            
            # 첫 번째 결과의 전체 문서 가져오기
            results = []
            for sr in search_results[:2]:
                title = sr.get('title', '').replace('Wikipedia: ', '')
                if title:
                    article = await self.mcp_client.get_wikipedia_article(title)
                    if article:
                        results.append(article)
            
            if not results:
                results = search_results
            
            logger.info(f"Wikipedia returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {str(e)}")
            return []
    
    async def _collect_from_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """ArXiv MCP로 학술 논문 검색"""
        try:
            results = await self.mcp_client.search_arxiv(query)
            logger.info(f"ArXiv returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"ArXiv search failed: {str(e)}")
            return []
    
    async def _collect_from_web_scraper(self, query: str) -> List[Dict[str, Any]]:
        """웹 검색 후 상세 페이지 스크래핑"""
        try:
            # 먼저 DuckDuckGo로 검색
            search_results = await self.mcp_client.search_web(query)
            
            if not search_results:
                return []
            
            # 상위 결과 URL들을 스크래핑
            scraped_results = []
            for result in search_results[:3]:
                url = result.get('url', '')
                if url and url.startswith('http'):
                    scraped = await self.mcp_client.fetch_url(url)
                    if scraped:
                        scraped_results.append(scraped)
            
            logger.info(f"Web scraper returned {len(scraped_results)} results")
            return scraped_results if scraped_results else search_results
            
        except Exception as e:
            logger.error(f"Web scraper failed: {str(e)}")
            return []

    def _evaluate_evidence_quality(self, evidence_list: List[Dict[str, Any]]) -> float:
        """근거 품질 종합 평가"""
        if not evidence_list:
            return 0.0
        
        total_score = 0.0
        for evidence in evidence_list:
            reliability = evidence.get('reliability_score', 0.5)
            relevance = evidence.get('relevance_score', 0.5)
            
            source_weight = {
                'wikipedia': 1.0,
                'arxiv': 0.9,
                'duckduckgo': 0.8,
                'web_scraper': 0.7
            }.get(evidence.get('source', 'unknown'), 0.5)
            
            evidence_score = (reliability * 0.6 + relevance * 0.4) * source_weight
            total_score += evidence_score
        
        return min(1.0, total_score / len(evidence_list))
    
    async def _judge_claim_with_evidence(self, claim: str, evidence_list: List[Dict[str, Any]], 
                                        quality_score: float) -> float:
        """근거 기반 claim 점수 계산"""
        try:
            if not evidence_list:
                return 0.0
            
            # 1. claim에서 핵심 정보 추출
            claim_entities = await self._extract_key_entities(claim)
            
            # 2. 각 근거에서 핵심 정보 추출
            evidence_entities_list = []
            for evidence in evidence_list:
                evidence_entities = await self._extract_key_entities(evidence.get('content', ''))
                evidence_entities_list.append(evidence_entities)
            
            # 3. 시스템적 점수 계산
            score = self._systematic_verdict_calculation(claim, claim_entities, evidence_list, evidence_entities_list)
            
            logger.info(f"Claim score: {score:.3f}")
            return score
                
        except Exception as e:
            logger.error(f"Failed to judge claim: {str(e)}")
            return 0.0
    
    async def _extract_key_entities(self, text: str) -> Dict[str, Any]:
        """텍스트에서 핵심 정보 추출"""
        try:
            judge = self.context.get_judge()
            
            prompt = f"""
다음 텍스트에서 핵심 정보를 추출해주세요:

텍스트: {text[:2000]}

다음 형식으로 추출해주세요:
- 날짜: [YYYY-MM-DD 형식으로, 예: 2023-07-11]
- 숫자: [모든 숫자값, 예: 100, 50.5]  
- 인명: [사람 이름들]
- 회사명: [회사/조직 이름들]
- 제품명: [제품/서비스 이름들]
- 지명: [장소/국가 이름들]

해당 정보가 없으면 "없음"으로 표시하세요.
"""
            
            result = await judge.analyze_text(prompt)
            
            entities = {
                'dates': [],
                'numbers': [],
                'persons': [],
                'companies': [],
                'products': [],
                'locations': []
            }
            
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- 날짜:'):
                    dates = line.replace('- 날짜:', '').strip()
                    if dates != '없음':
                        entities['dates'] = [d.strip() for d in dates.split(',')]
                elif line.startswith('- 숫자:'):
                    numbers = line.replace('- 숫자:', '').strip()
                    if numbers != '없음':
                        entities['numbers'] = [n.strip() for n in numbers.split(',')]
                elif line.startswith('- 인명:'):
                    persons = line.replace('- 인명:', '').strip()
                    if persons != '없음':
                        entities['persons'] = [p.strip() for p in persons.split(',')]
                elif line.startswith('- 회사명:'):
                    companies = line.replace('- 회사명:', '').strip()
                    if companies != '없음':
                        entities['companies'] = [c.strip() for c in companies.split(',')]
                elif line.startswith('- 제품명:'):
                    products = line.replace('- 제품명:', '').strip()
                    if products != '없음':
                        entities['products'] = [p.strip() for p in products.split(',')]
                elif line.startswith('- 지명:'):
                    locations = line.replace('- 지명:', '').strip()
                    if locations != '없음':
                        entities['locations'] = [l.strip() for l in locations.split(',')]
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {str(e)}")
            return {'dates': [], 'numbers': [], 'persons': [], 'companies': [], 'products': [], 'locations': []}
    
    def _systematic_verdict_calculation(self, claim: str, claim_entities: Dict[str, Any], 
                                      evidence_list: List[Dict[str, Any]], 
                                      evidence_entities_list: List[Dict[str, Any]]) -> float:
        """비례적 점수 체계로 점수 계산"""
        
        if not evidence_list or not any(evidence.get('content', '').strip() for evidence in evidence_list):
            return 0.0
        
        total_claim_elements = 0
        supported_elements = 0
        conflicted_elements = 0
        
        for entity_type in ['dates', 'numbers', 'persons', 'companies', 'products', 'locations']:
            claim_items = claim_entities.get(entity_type, [])
            
            if not claim_items:
                continue
                
            total_claim_elements += len(claim_items)
            
            for claim_item in claim_items:
                claim_item_clean = claim_item.lower().strip()
                
                found_support = False
                found_conflict = False
                
                for evidence_entities in evidence_entities_list:
                    evidence_items = evidence_entities.get(entity_type, [])
                    
                    exact_match = any(claim_item_clean == evidence_item.lower().strip() 
                                    for evidence_item in evidence_items)
                    
                    if exact_match:
                        found_support = True
                        break
                    elif evidence_items:
                        found_conflict = True
                
                if found_support:
                    supported_elements += 1
                elif found_conflict:
                    conflicted_elements += 1
        
        if conflicted_elements > 0:
            return 0.0
        
        if total_claim_elements == 0:
            return 0.5
        
        if supported_elements == total_claim_elements:
            return 1.0
        
        partial_score = supported_elements / total_claim_elements
        
        logger.info(f"Score calculation - Total: {total_claim_elements}, Supported: {supported_elements}, Conflicted: {conflicted_elements}, Score: {partial_score:.3f}")
        
        return partial_score
    
    async def _generate_search_query(self, claim: str) -> str:
        """Claim에서 검색 쿼리 생성"""
        try:
            judge = self.context.get_judge()
            
            prompt = f"""
다음 claim을 검증하기 위한 최적의 검색 쿼리를 생성해주세요:

Claim: {claim}

핵심 키워드와 검색어를 포함한 효과적인 검색 쿼리를 반환하세요.
예시: "OpenAI GPT-4 release date 2023" 또는 "Tesla stock price 2024"

검색 쿼리만 반환하세요 (설명 없이):"""
            
            result = await judge.analyze_text(prompt)
            query = result.strip()
            
            # 따옴표 제거
            query = query.strip('"\'')
            
            return query if query else ' '.join(claim.split()[:5])
            
        except Exception as e:
            logger.error(f"Failed to generate search query: {str(e)}")
            return ' '.join(claim.split()[:5])
