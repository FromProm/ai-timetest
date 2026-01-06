import asyncio
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Optional

from app.core.schemas import (
    JobCreateRequest, JobResponse, JobListResponse, JobStatus
)
from app.orchestrator.context import ExecutionContext
from app.orchestrator.pipeline import Orchestrator
from app.core.errors import PromptEvalError, ErrorCategory
from app.core.logging import get_structured_logger

logger = logging.getLogger(__name__)
structured_logger = get_structured_logger(__name__)
router = APIRouter(tags=["jobs"])

# 재시도 설정
MAX_RETRY_COUNT = 3

# Context will be injected from main.py
def get_context():
    from app.main import context
    return context

# Context initialization will be handled in main.py

@router.post("/jobs", response_model=JobResponse)
async def create_job(request: JobCreateRequest, background_tasks: BackgroundTasks):
    """프롬프트 평가 작업 생성"""
    try:
        # 작업 생성
        context = get_context()
        storage = context.get_storage()
        job_id = await storage.create_job(request.model_dump())
        
        structured_logger.info(
            "Job created",
            request_id=job_id,
            stage="job_creation",
            metadata={"prompt_type": request.prompt_type.value}
        )
        
        # 백그라운드에서 실행
        background_tasks.add_task(run_evaluation, job_id, request, retry_count=0)
        
        # 생성된 작업 반환
        job = await storage.get_job(job_id)
        return job
        
    except Exception as e:
        structured_logger.error(
            f"Job creation failed: {str(e)}",
            stage="job_creation",
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """작업 조회 - 완료될 때까지 대기"""
    try:
        context = get_context()
        storage = context.get_storage()
        
        # 작업이 완료될 때까지 폴링
        max_wait_time = 300  # 5분 최대 대기
        poll_interval = 2    # 2초마다 확인
        waited_time = 0
        
        while waited_time < max_wait_time:
            job = await storage.get_job(job_id)
            
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            
            # 완료되었거나 실패한 경우 즉시 반환
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                return job
            
            # 아직 실행 중이면 잠시 대기
            await asyncio.sleep(poll_interval)
            waited_time += poll_interval
        
        # 타임아웃된 경우 현재 상태 반환
        job = await storage.get_job(job_id)
        return job
        
    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(
            f"Job retrieval failed: {str(e)}",
            request_id=job_id,
            stage="job_retrieval",
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100)
):
    """작업 목록 조회"""
    try:
        context = get_context()
        storage = context.get_storage()
        jobs = await storage.list_jobs(page, size)
        total = await storage.count_jobs()
        
        return JobListResponse(
            jobs=jobs,
            total=total,
            page=page,
            size=size
        )
        
    except Exception as e:
        structured_logger.error(
            f"Job listing failed: {str(e)}",
            stage="job_listing",
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jobs/{job_id}/rerun", response_model=JobResponse)
async def rerun_job(job_id: str, background_tasks: BackgroundTasks):
    """작업 재실행"""
    try:
        context = get_context()
        storage = context.get_storage()
        job = await storage.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # 새 요청 객체 생성
        request = JobCreateRequest(
            prompt=job.prompt,
            example_inputs=job.example_inputs,
            prompt_type=job.prompt_type,
            recommended_model=job.recommended_model,
            repeat_count=job.repeat_count
        )
        
        # 상태 초기화
        await storage.update_job(job_id, {
            'status': JobStatus.PENDING,
            'result': None,
            'error_message': None
        })
        
        structured_logger.info(
            "Job rerun initiated",
            request_id=job_id,
            stage="job_rerun"
        )
        
        # 백그라운드에서 재실행
        background_tasks.add_task(run_evaluation, job_id, request, retry_count=0)
        
        # 업데이트된 작업 반환
        updated_job = await storage.get_job(job_id)
        return updated_job
        
    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(
            f"Job rerun failed: {str(e)}",
            request_id=job_id,
            stage="job_rerun",
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))

async def run_evaluation(job_id: str, request: JobCreateRequest, retry_count: int = 0):
    """백그라운드 평가 실행 (재시도 로직 포함)"""
    context = get_context()
    storage = context.get_storage()
    
    try:
        # 상태 업데이트: RUNNING
        await storage.update_job(job_id, {'status': JobStatus.RUNNING})
        
        structured_logger.info(
            "Evaluation started",
            request_id=job_id,
            stage="pipeline_execution",
            retry_count=retry_count,
            metadata={"prompt_type": request.prompt_type.value}
        )
        
        # 파이프라인 실행
        orchestrator = Orchestrator(context)
        result = await orchestrator.run(request)
        
        # 상태 업데이트: COMPLETED (결과 + 실행 결과 모두 저장)
        await storage.update_job(job_id, {
            'status': JobStatus.COMPLETED,
            'result': result,
            'execution_results': getattr(orchestrator, '_last_execution_results', None)
        })
        
        structured_logger.info(
            "Job completed successfully",
            request_id=job_id,
            stage="pipeline_execution"
        )
        
        # 자동 저장: S3/DynamoDB (또는 로컬 파일)
        try:
            job = await storage.get_job(job_id)
            if job:
                # title은 request에서 가져오거나 기본값 사용
                title = request.title or f"Prompt_{job_id[:8]}"
                description = request.description
                user_id = request.user_id
                
                save_result = await storage.save_completed_job(
                    job=job,
                    title=title,
                    description=description,
                    user_id=user_id
                )
                
                structured_logger.info(
                    "Job auto-saved to storage",
                    request_id=job_id,
                    stage="auto_save",
                    metadata=save_result
                )
        except Exception as save_error:
            # 저장 실패해도 평가 결과는 유지
            structured_logger.warning(
                f"Auto-save failed (job still completed): {str(save_error)}",
                request_id=job_id,
                stage="auto_save",
                error_type=type(save_error).__name__
            )
        
    except PromptEvalError as e:
        # 구조화된 에러 처리
        structured_logger.error(
            f"Job failed: {e.message}",
            request_id=job_id,
            stage="pipeline_execution",
            error_type=type(e).__name__,
            retry_count=retry_count,
            metadata=e.metadata
        )
        
        # 재시도 가능한 에러이고 재시도 횟수가 남았으면 재시도
        if e.category == ErrorCategory.RETRYABLE and retry_count < MAX_RETRY_COUNT:
            structured_logger.warning(
                f"Retrying job (attempt {retry_count + 1}/{MAX_RETRY_COUNT})",
                request_id=job_id,
                stage="retry",
                retry_count=retry_count + 1
            )
            # 잠시 대기 후 재시도 (exponential backoff)
            await asyncio.sleep(2 ** retry_count)
            await run_evaluation(job_id, request, retry_count + 1)
            return
        
        # 재시도 불가 또는 재시도 횟수 초과 → FAILED
        await storage.update_job(job_id, {
            'status': JobStatus.FAILED,
            'error_message': f"[{e.category.value}] {e.message}"
        })
        
    except Exception as e:
        # 예상치 못한 에러
        structured_logger.error(
            f"Unexpected error: {str(e)}",
            request_id=job_id,
            stage="pipeline_execution",
            error_type=type(e).__name__,
            retry_count=retry_count
        )
        
        # 상태 업데이트: FAILED
        await storage.update_job(job_id, {
            'status': JobStatus.FAILED,
            'error_message': str(e)
        })


@router.get("/jobs/{job_id}/dynamodb", response_model=None)
async def get_job_dynamodb_format(
    job_id: str,
    title: str = Query(..., description="프롬프트 제목"),
    description: Optional[str] = Query(None, description="프롬프트 설명"),
    user_id: Optional[str] = Query(None, description="사용자 ID (SB에서 전달)"),
    s3_bucket: Optional[str] = Query(None, description="S3 버킷명")
):
    """작업 결과를 DynamoDB 저장 형식으로 변환하여 반환"""
    from app.core.schemas import convert_job_to_dynamodb_record
    
    try:
        context = get_context()
        storage = context.get_storage()
        job = await storage.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400, 
                detail=f"Job is not completed yet. Current status: {job.status.value}"
            )
        
        # DynamoDB 형식으로 변환
        dynamodb_record = convert_job_to_dynamodb_record(
            job=job,
            title=title,
            description=description,
            user_id=user_id,
            s3_bucket=s3_bucket
        )
        
        structured_logger.info(
            "DynamoDB record generated",
            request_id=job_id,
            stage="dynamodb_conversion",
            metadata={"user_id": user_id, "title": title}
        )
        
        # alias 사용해서 PK, SK 등으로 출력
        return dynamodb_record.model_dump(by_alias=True)
        
    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(
            f"DynamoDB conversion failed: {str(e)}",
            request_id=job_id,
            stage="dynamodb_conversion",
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/s3-examples", response_model=None)
async def get_job_s3_examples_format(job_id: str):
    """작업 결과를 S3 저장용 예시 데이터 형식으로 반환"""
    from app.core.schemas import create_s3_examples_data
    
    try:
        context = get_context()
        storage = context.get_storage()
        job = await storage.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400, 
                detail=f"Job is not completed yet. Current status: {job.status.value}"
            )
        
        # S3 예시 데이터 생성
        s3_data = create_s3_examples_data(job)
        
        structured_logger.info(
            "S3 examples data generated",
            request_id=job_id,
            stage="s3_conversion"
        )
        
        return s3_data.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(
            f"S3 examples conversion failed: {str(e)}",
            request_id=job_id,
            stage="s3_conversion",
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))
