import random
import json
from app.adapters.judge.base import BaseJudge

class MockJudge(BaseJudge):
    """테스트용 Mock Judge - 실제 AI 호출 없이 가짜 판별 결과 생성"""
    
    async def judge_factuality(self, question: str, answer: str) -> bool:
        """가짜 사실성 판별"""
        
        # 간단한 휴리스틱으로 판별
        factual_keywords = [
            "약", "대략", "정도", "추정", "million", "billion", 
            "미터", "km", "년", "명", "개", "번"
        ]
        
        hallucination_keywords = [
            "확실하지", "모르겠", "어렵습니다", "찾아서", "구체적인"
        ]
        
        # 환각 키워드가 있으면 FALSE
        if any(keyword in answer for keyword in hallucination_keywords):
            return False
        
        # 사실적 키워드가 있으면 TRUE
        if any(keyword in answer for keyword in factual_keywords):
            return True
        
        # 그 외에는 80% 확률로 TRUE (대부분 사실적이라고 가정)
        return random.random() > 0.2
    
    async def evaluate(self, prompt: str, task_type: str = "general") -> str:
        """가짜 AI 평가 메서드"""
        
        if task_type == "condition_extraction":
            # 실제 프롬프트에서 조건 추출 시뮬레이션
            conditions = []
            direction = ""
            
            # 프롬프트 분석하여 실제 조건 추출
            if "정확한 사실과 근거" in prompt:
                conditions.append("조건1: 정확한 사실 정보 제공")
                conditions.append("조건2: 근거 기반 답변")
                direction = "사실적이고 근거가 있는 정보 제공"
            elif "창의적" in prompt or "소설" in prompt:
                conditions.append("조건1: 창의적 내용 작성")
                conditions.append("조건2: 흥미로운 스토리")
                if "500자" in prompt:
                    conditions.append("조건3: 500자 내외 길이")
                direction = "창의적이고 흥미로운 창작물 생성"
            elif "이미지" in prompt:
                conditions.append("조건1: 아름다운 이미지 생성")
                conditions.append("조건2: 창의적 시각 표현")
                direction = "설명에 맞는 창의적 이미지 생성"
            else:
                # 기본 조건들
                conditions = [
                    "조건1: 질문에 적절한 답변",
                    "조건2: 명확하고 이해하기 쉬운 설명"
                ]
                direction = "질문에 대한 적절하고 유용한 답변 제공"
            
            return json.dumps({
                "explicit_conditions": conditions,
                "direction": direction
            }, ensure_ascii=False)
        
        elif task_type == "compliance_evaluation":
            # 준수 평가 가짜 응답 - 출력 내용에 따라 다르게 평가
            if "OpenAI" in prompt and ("2023년 3월" in prompt or "GPT-4" in prompt):
                # 정확한 정보가 포함된 경우
                compliance_status = "지킴"
                reason = "정확한 날짜와 사실 정보 포함"
            elif "노벨 물리학상" in prompt and ("홉필드" in prompt or "힌턴" in prompt):
                # 정확한 정보가 포함된 경우  
                compliance_status = "지킴"
                reason = "정확한 수상자 정보 포함"
            elif "윤석열" in prompt and "2022년 5월" in prompt:
                # 정확한 정보가 포함된 경우
                compliance_status = "지킴" 
                reason = "정확한 대통령 정보와 취임일 포함"
            elif "죄송합니다" in prompt or "어렵습니다" in prompt or "구체적인" in prompt:
                # 모호한 답변인 경우
                compliance_status = "안지킴"
                reason = "구체적인 답변을 제공하지 않음"
            else:
                # 기타 경우
                compliance_status = random.choice(["지킴", "애매함"])
                reason = "부분적으로 조건 충족"
            
            return json.dumps({
                "explicit_conditions_compliance": [
                    {"condition": "조건1", "status": compliance_status, "reason": reason},
                    {"condition": "조건2", "status": compliance_status, "reason": reason}
                ],
                "direction_compliance": {"status": compliance_status, "reason": f"방향성 {compliance_status} - {reason}"},
                "overall_assessment": f"전체적으로 {compliance_status} 상태입니다."
            }, ensure_ascii=False)
        
    async def evaluate_image(self, prompt: str, image_base64: str) -> str:
        """가짜 이미지 평가 메서드 (VLM 시뮬레이션)"""
        import json
        
        # 이미지가 유효한지 간단히 확인
        is_valid_image = (
            image_base64 and 
            len(image_base64) > 100 and
            (image_base64.startswith('/9j/') or image_base64.startswith('iVBORw0KGgo'))
        )
        
        if not is_valid_image:
            return json.dumps({
                "explicit_conditions_compliance": [],
                "direction_compliance": {"status": "안지킴", "reason": "유효한 이미지가 아님"},
                "overall_assessment": "이미지 생성 실패"
            }, ensure_ascii=False)
        
        # Mock 평가 결과 생성
        status = random.choice(["지킴", "지킴", "애매함"])  # 대부분 지킴으로
        
        return json.dumps({
            "explicit_conditions_compliance": [
                {"condition": "이미지 생성", "status": status, "reason": "이미지가 정상적으로 생성됨"},
                {"condition": "프롬프트 반영", "status": status, "reason": "프롬프트 내용이 이미지에 반영됨"}
            ],
            "direction_compliance": {"status": status, "reason": "전체적인 방향성 준수"},
            "overall_assessment": f"이미지가 프롬프트 요구사항을 {status} 상태입니다."
        }, ensure_ascii=False)
    
    async def analyze_text(self, prompt: str) -> str:
        """가짜 텍스트 분석 메서드"""
        
        # Claim type 분류 요청인 경우
        if "FACT_VERIFIABLE" in prompt and "타입" in prompt:
            # 실제 출력에서 FACT_VERIFIABLE 문장들 추출 시뮬레이션
            if "OpenAI" in prompt and "GPT-4" in prompt:
                mock_claims = [
                    "OpenAI는 2023년 3월 14일에 GPT-4를 공식 발표했습니다.",
                    "GPT-4는 이전 버전보다 향상된 추론 능력과 창의성을 보여줍니다.",
                    "GPT-4는 다양한 전문 시험에서 인간 수준의 성능을 달성했습니다."
                ]
            elif "노벨 물리학상" in prompt and "2024" in prompt:
                mock_claims = [
                    "2024년 노벨 물리학상은 존 홉필드와 제프리 힌턴이 공동 수상했습니다.",
                    "이들은 인공신경망의 기초가 되는 기계학습 방법을 개발한 공로를 인정받았습니다.",
                    "힌턴은 딥러닝의 아버지로 불립니다."
                ]
            elif "한국" in prompt and "대통령" in prompt:
                mock_claims = [
                    "한국의 현재 대통령은 윤석열입니다.",
                    "그는 2022년 5월 10일에 제20대 대통령으로 취임했습니다.",
                    "윤석열 대통령은 이전에 검찰총장을 역임했습니다."
                ]
            else:
                mock_claims = [
                    "해당 질문에 대한 정확한 답변을 드리기 어렵습니다.",
                    "더 구체적인 질문을 해주시면 도움을 드릴 수 있습니다."
                ]
            return "\n".join(mock_claims)
        
        # 핵심 정보 추출 요청인 경우
        elif "핵심 정보를 추출" in prompt and "날짜:" in prompt:
            # 텍스트에서 실제 정보 추출 시뮬레이션
            text = prompt.split("텍스트:")[1].split("다음 형식으로")[0].strip() if "텍스트:" in prompt else ""
            
            # 실제 텍스트 분석하여 정보 추출
            dates = []
            numbers = []
            persons = []
            companies = []
            products = []
            locations = []
            
            # 날짜 추출
            if "2023년 3월 14일" in text or "2023-03-14" in text:
                dates.append("2023-03-14")
            elif "2022년 5월 10일" in text or "2022-05-10" in text:
                dates.append("2022-05-10")
            elif "2024년" in text:
                dates.append("2024-01-01")
            
            # 인명 추출
            if "윤석열" in text:
                persons.append("윤석열")
            if "홉필드" in text or "Hopfield" in text:
                persons.append("존 홉필드")
            if "힌턴" in text or "Hinton" in text:
                persons.append("제프리 힌턴")
            
            # 회사명 추출
            if "OpenAI" in text:
                companies.append("OpenAI")
            
            # 제품명 추출
            if "GPT-4" in text:
                products.append("GPT-4")
            
            # 지명 추출
            if "한국" in text:
                locations.append("한국")
            
            # 결과 포맷팅
            result = f"""- 날짜: {', '.join(dates) if dates else '없음'}
- 숫자: {', '.join(numbers) if numbers else '없음'}
- 인명: {', '.join(persons) if persons else '없음'}
- 회사명: {', '.join(companies) if companies else '없음'}
- 제품명: {', '.join(products) if products else '없음'}
- 지명: {', '.join(locations) if locations else '없음'}"""
            
            return result
        
        # MCP 선택 요청인 경우
        elif "MCP 타입" in prompt and "선택" in prompt:
            return random.choice(["SEARCH", "CRAWLING", "OFFICIAL_DATA", "ACADEMIC"])
        
        # Verdict 판정 요청인 경우
        elif "SUPPORTED" in prompt and "REFUTED" in prompt:
            return random.choice(["SUPPORTED", "REFUTED", "INSUFFICIENT"])
        
        else:
            return "Mock 텍스트 분석 결과입니다."