from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import boto3
import json
from app.core.config import settings

router = APIRouter(tags=["debug"])

@router.get("/debug/s3/buckets")
async def list_s3_buckets():
    """S3 버킷 목록 확인"""
    try:
        s3_client = boto3.client(
            's3',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None
        )
        
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        
        return {
            "buckets": buckets,
            "target_bucket": settings.s3_bucket_name,
            "bucket_exists": settings.s3_bucket_name in buckets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 access failed: {str(e)}")

@router.get("/debug/s3/jobs")
async def list_s3_jobs():
    """S3에 저장된 작업 목록 확인"""
    try:
        s3_client = boto3.client(
            's3',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None
        )
        
        # jobs/ 폴더 내용 확인
        response = s3_client.list_objects_v2(
            Bucket=settings.s3_bucket_name,
            Prefix="jobs/",
            Delimiter="/"
        )
        
        job_folders = []
        for prefix in response.get('CommonPrefixes', []):
            job_id = prefix['Prefix'].split('/')[1]
            job_folders.append(job_id)
        
        return {
            "bucket": settings.s3_bucket_name,
            "job_count": len(job_folders),
            "job_ids": job_folders
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 jobs listing failed: {str(e)}")

@router.get("/debug/s3/jobs/{job_id}")
async def get_s3_job_files(job_id: str):
    """특정 작업의 S3 파일들 확인"""
    try:
        s3_client = boto3.client(
            's3',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None
        )
        
        # 해당 job의 모든 파일 확인
        response = s3_client.list_objects_v2(
            Bucket=settings.s3_bucket_name,
            Prefix=f"jobs/{job_id}/"
        )
        
        files = []
        for obj in response.get('Contents', []):
            files.append({
                "key": obj['Key'],
                "size": obj['Size'],
                "last_modified": obj['LastModified'].isoformat()
            })
        
        return {
            "job_id": job_id,
            "files": files,
            "file_count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 job files listing failed: {str(e)}")

@router.get("/debug/s3/jobs/{job_id}/metadata")
async def get_s3_job_metadata(job_id: str):
    """S3에서 작업 메타데이터 내용 확인"""
    try:
        s3_client = boto3.client(
            's3',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None
        )
        
        # 메타데이터 파일 읽기
        response = s3_client.get_object(
            Bucket=settings.s3_bucket_name,
            Key=f"jobs/{job_id}/metadata.json"
        )
        
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        
        return {
            "job_id": job_id,
            "metadata": metadata,
            "ai_outputs_stored": metadata.get('ai_outputs_stored', 'unknown')
        }
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Job {job_id} metadata not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {str(e)}")

@router.get("/debug/s3/jobs/{job_id}/result")
async def get_s3_job_result(job_id: str):
    """S3에서 평가 결과 확인"""
    try:
        s3_client = boto3.client(
            's3',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None
        )
        
        # 평가 결과 파일 읽기
        response = s3_client.get_object(
            Bucket=settings.s3_bucket_name,
            Key=f"jobs/{job_id}/evaluation_result.json"
        )
        
        result = json.loads(response['Body'].read().decode('utf-8'))
        
        return {
            "job_id": job_id,
            "evaluation_result": result,
            "contains_ai_outputs": "outputs" in str(result)
        }
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Job {job_id} result not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {str(e)}")

@router.get("/debug/storage/backend")
async def get_storage_backend():
    """현재 사용 중인 저장소 백엔드 확인"""
    return {
        "storage_backend": settings.storage_backend,
        "s3_bucket_name": settings.s3_bucket_name,
        "table_name": settings.table_name,
        "mock_mode": settings.mock_mode,
        "database_url": settings.database_url
    }

@router.get("/debug/dynamodb/jobs/{job_id}/inputs")
async def get_job_inputs_from_s3(job_id: str):
    """S3에서 작업 입력 데이터 확인"""
    try:
        from app.main import context
        storage = context.get_storage()
        
        if hasattr(storage, 'get_job_inputs'):
            inputs = await storage.get_job_inputs(job_id)
            return {
                "job_id": job_id,
                "inputs": inputs,
                "has_inputs": inputs is not None
            }
        else:
            return {"error": "Storage backend does not support direct input access"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get inputs: {str(e)}")

@router.get("/debug/dynamodb/jobs/{job_id}/outputs")
async def get_job_outputs_from_s3(job_id: str):
    """S3에서 작업 출력 데이터 확인"""
    try:
        from app.main import context
        storage = context.get_storage()
        
        if hasattr(storage, 'get_job_outputs'):
            outputs = await storage.get_job_outputs(job_id)
            return {
                "job_id": job_id,
                "outputs": outputs,
                "has_outputs": outputs is not None,
                "contains_ai_generated_text": "execution_results" in str(outputs) if outputs else False
            }
        else:
            return {"error": "Storage backend does not support direct output access"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get outputs: {str(e)}")


@router.post("/debug/prompt/preview")
async def preview_filled_prompt(
    prompt: str,
    example_input: str
):
    """프롬프트에 입력이 어떻게 채워지는지 미리보기"""
    import json
    import re
    
    result = prompt
    has_placeholder = bool(re.search(r'\{\{.*?\}\}', prompt))
    
    # 1. JSON 파싱 시도
    try:
        data = json.loads(example_input)
        if isinstance(data, dict):
            for key, value in data.items():
                result = result.replace(f"{{{{{key}}}}}", str(value))
    except (json.JSONDecodeError, TypeError):
        pass
    
    # 2. 기본 플레이스홀더 치환
    result = result.replace("{{}}", example_input).replace("{{input}}", example_input)
    
    # 3. 플레이스홀더 없으면 맨 뒤에 추가
    if not has_placeholder:
        result = f"{result}\n\n{example_input}"
    
    return {
        "original_prompt": prompt,
        "example_input": example_input,
        "has_placeholder": has_placeholder,
        "filled_prompt": result
    }

@router.get("/debug/jobs/{job_id}/prompts")
async def get_job_filled_prompts(job_id: str):
    """작업에서 실제 LLM에 전달된 프롬프트들 확인"""
    try:
        from app.main import context
        storage = context.get_storage()
        job = await storage.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if not job.result or not job.result.execution_results:
            return {
                "job_id": job_id,
                "status": job.status.value,
                "message": "No execution results yet",
                "filled_prompts": []
            }
        
        filled_prompts = []
        executions = job.result.execution_results.get('executions', [])
        
        for exec_data in executions:
            filled_prompts.append({
                "input_index": exec_data.get('input_index'),
                "input_content": exec_data.get('input_content'),
                "filled_prompt": exec_data.get('filled_prompt', 'N/A - 이전 버전에서 실행됨'),
                "model": exec_data.get('model')
            })
        
        return {
            "job_id": job_id,
            "original_prompt": job.prompt,
            "filled_prompts": filled_prompts
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prompts: {str(e)}")


# ============================================
# 이미지 미리보기 엔드포인트
# ============================================

from fastapi.responses import HTMLResponse, Response
import base64

@router.get("/debug/jobs/{job_id}/images", response_class=HTMLResponse)
async def preview_job_images(job_id: str):
    """작업에서 생성된 이미지들을 HTML 페이지로 미리보기"""
    try:
        from app.main import context
        from app.core.schemas import PromptType
        
        storage = context.get_storage()
        job = await storage.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.prompt_type != PromptType.TYPE_B_IMAGE:
            return HTMLResponse(content=f"""
            <html>
            <head><title>Not an Image Job</title></head>
            <body>
                <h1>⚠️ 이미지 작업이 아닙니다</h1>
                <p>Job ID: {job_id}</p>
                <p>Prompt Type: {job.prompt_type.value}</p>
                <p>이 엔드포인트는 type_b_image 작업에서만 사용 가능합니다.</p>
            </body>
            </html>
            """)
        
        if not job.result or not job.result.execution_results:
            return HTMLResponse(content=f"""
            <html>
            <head><title>No Results</title></head>
            <body>
                <h1>⏳ 결과 없음</h1>
                <p>Job ID: {job_id}</p>
                <p>Status: {job.status.value}</p>
                <p>아직 실행 결과가 없습니다.</p>
            </body>
            </html>
            """)
        
        # 이미지 추출
        executions = job.result.execution_results.get('executions', [])
        
        html_content = f"""
        <html>
        <head>
            <title>Image Preview - {job_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; }}
                .info {{ background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .input-group {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .input-group h2 {{ color: #666; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                .images {{ display: flex; flex-wrap: wrap; gap: 15px; }}
                .image-card {{ background: #fafafa; padding: 10px; border-radius: 8px; text-align: center; }}
                .image-card img {{ max-width: 300px; max-height: 300px; border-radius: 4px; }}
                .image-card p {{ margin: 10px 0 0 0; color: #666; font-size: 14px; }}
                .no-image {{ color: #999; padding: 50px; background: #eee; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <h1>🖼️ 이미지 미리보기</h1>
            <div class="info">
                <p><strong>Job ID:</strong> {job_id}</p>
                <p><strong>Prompt:</strong> {job.prompt[:200]}{'...' if len(job.prompt) > 200 else ''}</p>
                <p><strong>Status:</strong> {job.status.value}</p>
            </div>
        """
        
        for exec_data in executions:
            input_idx = exec_data.get('input_index', 0)
            outputs = exec_data.get('outputs', [])
            input_content = exec_data.get('input_content', '')[:100]
            
            html_content += f"""
            <div class="input-group">
                <h2>입력 #{input_idx + 1}: {input_content}{'...' if len(exec_data.get('input_content', '')) > 100 else ''}</h2>
                <div class="images">
            """
            
            for out_idx, output in enumerate(outputs):
                if output and len(output) > 100:
                    # base64 이미지로 가정
                    # 이미지 타입 감지
                    if output.startswith('/9j/'):
                        mime = 'image/jpeg'
                    else:
                        mime = 'image/png'
                    
                    html_content += f"""
                    <div class="image-card">
                        <img src="data:{mime};base64,{output}" alt="Output {out_idx + 1}">
                        <p>출력 #{out_idx + 1}</p>
                    </div>
                    """
                else:
                    html_content += f"""
                    <div class="image-card">
                        <div class="no-image">이미지 없음</div>
                        <p>출력 #{out_idx + 1}: {output[:50] if output else 'Empty'}</p>
                    </div>
                    """
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview images: {str(e)}")


@router.get("/debug/jobs/{job_id}/images/{input_index}/{output_index}")
async def get_single_image(job_id: str, input_index: int, output_index: int):
    """특정 이미지를 직접 반환 (PNG/JPEG)"""
    try:
        from app.main import context
        from app.core.schemas import PromptType
        
        storage = context.get_storage()
        job = await storage.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.prompt_type != PromptType.TYPE_B_IMAGE:
            raise HTTPException(status_code=400, detail="Not an image generation job")
        
        if not job.result or not job.result.execution_results:
            raise HTTPException(status_code=404, detail="No execution results")
        
        executions = job.result.execution_results.get('executions', [])
        
        # 해당 입력 찾기
        exec_data = None
        for e in executions:
            if e.get('input_index') == input_index:
                exec_data = e
                break
        
        if not exec_data:
            raise HTTPException(status_code=404, detail=f"Input {input_index} not found")
        
        outputs = exec_data.get('outputs', [])
        if output_index >= len(outputs):
            raise HTTPException(status_code=404, detail=f"Output {output_index} not found")
        
        output = outputs[output_index]
        if not output or len(output) < 100:
            raise HTTPException(status_code=404, detail="No valid image data")
        
        # base64 디코딩
        image_data = base64.b64decode(output)
        
        # 이미지 타입 감지
        if output.startswith('/9j/'):
            media_type = 'image/jpeg'
        else:
            media_type = 'image/png'
        
        return Response(content=image_data, media_type=media_type)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get image: {str(e)}")
