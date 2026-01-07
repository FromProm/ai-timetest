import logging
import asyncio
from typing import Dict, Any, List
from app.orchestrator.context import ExecutionContext
from app.core.schemas import ExampleInput, PromptType

logger = logging.getLogger(__name__)

class EmbedStage:
    """임베딩 생성 단계"""
    
    def __init__(self, context: ExecutionContext):
        self.context = context
    
    async def execute(
        self, 
        execution_results: Dict[str, Any], 
        example_inputs: List[ExampleInput],
        prompt_type: PromptType
    ) -> Dict[str, Any]:
        """
        임베딩 생성
        - 입력과 출력을 각각 임베딩
        - 텍스트/이미지 타입에 따라 적절한 모델 선택
        """
        logger.info("Generating embeddings for inputs and outputs")
        
        try:
            embedder = self.context.get_embedder()
            executions = execution_results['executions']
            
            # 입력 임베딩 생성
            input_embeddings = await self._embed_inputs(embedder, example_inputs)
            
            # 출력 임베딩 생성
            output_embeddings = await self._embed_outputs(embedder, executions, prompt_type)
            
            return {
                'inputs': input_embeddings,
                'outputs': output_embeddings
            }
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    async def _embed_inputs(self, embedder, example_inputs: List[ExampleInput]) -> List[Dict[str, Any]]:
        """입력 임베딩 생성 (병렬 처리)"""
        
        async def embed_single_input(i: int, example_input: ExampleInput) -> Dict[str, Any]:
            """단일 입력 임베딩"""
            logger.info(f"Embedding input {i+1}/{len(example_inputs)}")
            
            if example_input.input_type == "image":
                # 이미지 입력 임베딩 (Titan Multimodal + Cohere v4) - 병렬
                try:
                    titan_task = embedder.embed_image_titan(example_input.content)
                    cohere_task = embedder.embed_image_cohere(example_input.content)
                    titan_emb, cohere_emb = await asyncio.gather(titan_task, cohere_task)
                    
                    return {
                        'index': i,
                        'content': example_input.content[:100] + '...',
                        'type': example_input.input_type,
                        'titan_embedding': titan_emb,
                        'cohere_embedding': cohere_emb
                    }
                except Exception as e:
                    logger.error(f"Image input embedding failed: {str(e)}")
                    return {
                        'index': i,
                        'content': example_input.content[:100] + '...',
                        'type': example_input.input_type,
                        'titan_embedding': None,
                        'cohere_embedding': None
                    }
            else:
                # 텍스트 입력 임베딩 (Titan Text + Cohere Multilingual) - 병렬
                titan_task = embedder.embed_text(example_input.content)
                cohere_task = embedder.embed_multilingual(example_input.content)
                titan_emb, cohere_emb = await asyncio.gather(titan_task, cohere_task)
                
                return {
                    'index': i,
                    'content': example_input.content,
                    'type': example_input.input_type,
                    'titan_embedding': titan_emb,
                    'cohere_embedding': cohere_emb
                }
        
        # 모든 입력을 병렬로 임베딩
        tasks = [embed_single_input(i, inp) for i, inp in enumerate(example_inputs)]
        results = await asyncio.gather(*tasks)
        
        # 인덱스 순서대로 정렬
        return sorted(results, key=lambda x: x['index'])
    
    async def _embed_outputs(self, embedder, executions: List[Dict], prompt_type: PromptType) -> List[Dict[str, Any]]:
        """출력 임베딩 생성 (병렬 처리)"""
        
        async def embed_single_output(output_idx: int, output: str, is_image_type: bool) -> Dict[str, Any]:
            """단일 출력 임베딩"""
            if not output or not output.strip():
                return {
                    'output_index': output_idx,
                    'content': output,
                    'titan_embedding': None,
                    'cohere_embedding': None
                }
            
            if is_image_type:
                # 이미지 생성 타입
                if self._is_base64_image(output):
                    try:
                        titan_task = embedder.embed_image_titan(output)
                        cohere_task = embedder.embed_image_cohere(output)
                        titan_emb, cohere_emb = await asyncio.gather(titan_task, cohere_task)
                        
                        return {
                            'output_index': output_idx,
                            'content': output[:100] + '...',
                            'is_image': True,
                            'titan_embedding': titan_emb,
                            'cohere_embedding': cohere_emb
                        }
                    except Exception as e:
                        logger.error(f"Image output embedding failed: {str(e)}")
                        return {
                            'output_index': output_idx,
                            'content': output[:100] + '...',
                            'is_image': True,
                            'titan_embedding': None,
                            'cohere_embedding': None
                        }
                else:
                    logger.warning(f"Output {output_idx} is not a valid base64 image")
                    return {
                        'output_index': output_idx,
                        'content': output[:200],
                        'is_image': False,
                        'titan_embedding': None,
                        'cohere_embedding': None
                    }
            else:
                # 텍스트 출력 임베딩 - 병렬
                titan_task = embedder.embed_text(output)
                cohere_task = embedder.embed_multilingual(output)
                titan_emb, cohere_emb = await asyncio.gather(titan_task, cohere_task)
                
                return {
                    'output_index': output_idx,
                    'content': output,
                    'is_image': False,
                    'titan_embedding': titan_emb,
                    'cohere_embedding': cohere_emb
                }
        
        async def embed_execution_group(exec_data: Dict) -> Dict[str, Any]:
            """단일 실행 그룹의 모든 출력 임베딩"""
            input_index = exec_data['input_index']
            outputs = exec_data['outputs']
            is_image_type = prompt_type == PromptType.TYPE_B_IMAGE
            
            logger.info(f"Embedding outputs for input {input_index+1}")
            
            # 해당 입력의 모든 출력을 병렬로 임베딩
            tasks = [embed_single_output(idx, out, is_image_type) for idx, out in enumerate(outputs)]
            exec_embeddings = await asyncio.gather(*tasks)
            
            # output_index 순서대로 정렬
            exec_embeddings = sorted(exec_embeddings, key=lambda x: x['output_index'])
            
            return {
                'input_index': input_index,
                'embeddings': exec_embeddings
            }
        
        # 모든 실행 그룹을 병렬로 처리
        tasks = [embed_execution_group(exec_data) for exec_data in executions]
        results = await asyncio.gather(*tasks)
        
        # input_index 순서대로 정렬
        return sorted(results, key=lambda x: x['input_index'])
    
    def _is_base64_image(self, content: str) -> bool:
        """base64 이미지인지 확인"""
        if not content:
            return False
        
        # 일반적인 base64 이미지 패턴 확인
        # PNG: iVBORw0KGgo...
        # JPEG: /9j/4AAQ...
        content = content.strip()
        
        if content.startswith('iVBORw0KGgo'):  # PNG
            return True
        if content.startswith('/9j/'):  # JPEG
            return True
        
        # 최소 길이 확인 (이미지는 보통 길다)
        if len(content) > 1000:
            try:
                import base64
                base64.b64decode(content[:100])
                return True
            except:
                pass
        
        return False
