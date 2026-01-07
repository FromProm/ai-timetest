"""
MCP Client - 외부 MCP 서버들과 통신하는 클라이언트
지원 MCP:
- wikipedia-mcp: 위키피디아 검색
- arxiv-mcp-server: 학술 논문 검색
- duckduckgo-mcp-server: 웹 검색
- mcp-server-fetch: 웹 스크래핑

HTTP 기반 직접 API 호출 방식 사용 (MCP 서버 프로세스 관리 불필요)
"""

import logging
import asyncio
import aiohttp
import urllib.parse
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseMCPAdapter(ABC):
    """MCP 어댑터 기본 클래스"""
    
    @abstractmethod
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """검색 수행"""
        pass
    
    async def close(self):
        """리소스 정리"""
        pass


class WikipediaAdapter(BaseMCPAdapter):
    """Wikipedia API 직접 호출 어댑터"""
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.ko_base_url = "https://ko.wikipedia.org/w/api.php"
        # Wikipedia API는 적절한 User-Agent가 필수
        self.headers = {
            'User-Agent': 'PromptEvalBot/1.0 (Windows; Python/aiohttp) prompt-eval-service'
        }
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """위키피디아 검색"""
        try:
            # 한글이 포함되어 있으면 한국어 위키피디아 사용
            is_korean = any('\uac00' <= char <= '\ud7a3' for char in query)
            base_url = self.ko_base_url if is_korean else self.base_url
            
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': 3,
                'format': 'json',
                'utf8': 1,
                'origin': '*'
            }
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(base_url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status != 200:
                        logger.error(f"Wikipedia API error: {response.status}")
                        # Fallback: 직접 페이지 fetch
                        return await self._fallback_search(query, is_korean)
                    
                    data = await response.json()
                    search_results = data.get('query', {}).get('search', [])
                    
                    results = []
                    for item in search_results:
                        title = item.get('title', '')
                        snippet = item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                        
                        results.append({
                            'source': 'wikipedia',
                            'title': f'Wikipedia: {title}',
                            'url': f'{"https://ko.wikipedia.org" if is_korean else "https://en.wikipedia.org"}/wiki/{urllib.parse.quote(title)}',
                            'content': snippet,
                            'reliability_score': 0.95,
                            'relevance_score': 0.9
                        })
                    
                    return results
                    
        except asyncio.TimeoutError:
            logger.error("Wikipedia search timed out")
            return []
        except Exception as e:
            logger.error(f"Wikipedia search failed: {str(e)}")
            return []
    
    async def _fallback_search(self, query: str, is_korean: bool) -> List[Dict[str, Any]]:
        """API 실패시 직접 페이지 fetch로 fallback"""
        try:
            base = "https://ko.wikipedia.org" if is_korean else "https://en.wikipedia.org"
            url = f"{base}/wiki/{urllib.parse.quote(query)}"
            
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector, headers=self.headers) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        html = await response.text()
                        # 간단한 텍스트 추출
                        import re
                        # 첫 번째 문단 추출
                        content_match = re.search(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)
                        content = ''
                        if content_match:
                            content = re.sub(r'<[^>]+>', '', content_match.group(1))
                        
                        return [{
                            'source': 'wikipedia',
                            'title': f'Wikipedia: {query}',
                            'url': url,
                            'content': content[:1000] if content else f'Wikipedia article about {query}',
                            'reliability_score': 0.95,
                            'relevance_score': 0.85
                        }]
            return []
        except Exception as e:
            logger.error(f"Wikipedia fallback failed: {str(e)}")
            return []
    
    async def get_article(self, title: str) -> Optional[Dict[str, Any]]:
        """위키피디아 문서 전체 내용 가져오기"""
        try:
            is_korean = any('\uac00' <= char <= '\ud7a3' for char in title)
            base_url = self.ko_base_url if is_korean else self.base_url
            
            params = {
                'action': 'query',
                'titles': title,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                'format': 'json',
                'utf8': 1,
                'origin': '*'
            }
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(base_url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status != 200:
                        # Fallback
                        results = await self._fallback_search(title, is_korean)
                        return results[0] if results else None
                    
                    data = await response.json()
                    pages = data.get('query', {}).get('pages', {})
                    
                    for page_id, page_data in pages.items():
                        if page_id == '-1':
                            continue
                        
                        extract = page_data.get('extract', '')
                        return {
                            'source': 'wikipedia',
                            'title': page_data.get('title', title),
                            'content': extract[:5000],
                            'url': f'{"https://ko.wikipedia.org" if is_korean else "https://en.wikipedia.org"}/wiki/{urllib.parse.quote(title)}',
                            'reliability_score': 0.95,
                            'relevance_score': 0.9
                        }
                    
                    return None
                    
        except Exception as e:
            logger.error(f"Wikipedia get_article failed: {str(e)}")
            return None


class ArxivAdapter(BaseMCPAdapter):
    """ArXiv API 직접 호출 어댑터"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """ArXiv 논문 검색"""
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': 3,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(self.base_url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    if response.status != 200:
                        logger.error(f"ArXiv API error: {response.status}")
                        return []
                    
                    text = await response.text()
                    return self._parse_arxiv_response(text)
                    
        except asyncio.TimeoutError:
            logger.error("ArXiv search timed out")
            return []
        except Exception as e:
            logger.error(f"ArXiv search failed: {str(e)}")
            return []
    
    def _parse_arxiv_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """ArXiv XML 응답 파싱"""
        import re
        
        results = []
        
        # 간단한 XML 파싱 (정규식 사용)
        entries = re.findall(r'<entry>(.*?)</entry>', xml_text, re.DOTALL)
        
        for entry in entries:
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            id_match = re.search(r'<id>(.*?)</id>', entry)
            
            title = title_match.group(1).strip() if title_match else ''
            summary = summary_match.group(1).strip() if summary_match else ''
            arxiv_url = id_match.group(1).strip() if id_match else ''
            
            # 줄바꿈 정리
            title = ' '.join(title.split())
            summary = ' '.join(summary.split())
            
            if title:
                results.append({
                    'source': 'arxiv',
                    'title': f'ArXiv: {title}',
                    'url': arxiv_url,
                    'content': summary[:3000],
                    'reliability_score': 0.9,
                    'relevance_score': 0.85
                })
        
        return results
    
    async def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """ArXiv 논문 상세 정보"""
        try:
            params = {
                'id_list': arxiv_id,
                'max_results': 1
            }
            
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(self.base_url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    if response.status != 200:
                        return None
                    
                    text = await response.text()
                    results = self._parse_arxiv_response(text)
                    return results[0] if results else None
                    
        except Exception as e:
            logger.error(f"ArXiv get_paper failed: {str(e)}")
            return None


class DuckDuckGoAdapter(BaseMCPAdapter):
    """DuckDuckGo 검색 어댑터 - HTML 파싱 방식"""
    
    def __init__(self):
        self.search_url = "https://html.duckduckgo.com/html/"
        self.api_url = "https://api.duckduckgo.com/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """DuckDuckGo 검색 - HTML 파싱"""
        try:
            # 먼저 Instant Answer API 시도
            api_results = await self._search_api(query)
            if api_results:
                return api_results
            
            # API 실패시 HTML 검색
            return await self._search_html(query)
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []
    
    async def _search_api(self, query: str) -> List[Dict[str, Any]]:
        """DuckDuckGo Instant Answer API"""
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector, headers=self.headers) as session:
                async with session.get(self.api_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    results = []
                    
                    # Abstract
                    abstract = data.get('Abstract', '')
                    if abstract:
                        results.append({
                            'source': 'duckduckgo',
                            'title': data.get('Heading', query),
                            'url': data.get('AbstractURL', ''),
                            'content': abstract,
                            'reliability_score': 0.85,
                            'relevance_score': 0.9
                        })
                    
                    # Related Topics
                    for topic in data.get('RelatedTopics', [])[:3]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append({
                                'source': 'duckduckgo',
                                'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                                'url': topic.get('FirstURL', ''),
                                'content': topic.get('Text', ''),
                                'reliability_score': 0.75,
                                'relevance_score': 0.8
                            })
                    
                    return results
                    
        except Exception as e:
            logger.debug(f"DuckDuckGo API failed: {str(e)}")
            return []
    
    async def _search_html(self, query: str) -> List[Dict[str, Any]]:
        """DuckDuckGo HTML 검색 결과 파싱"""
        try:
            import re
            
            data = {'q': query}
            
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector, headers=self.headers) as session:
                async with session.post(self.search_url, data=data, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status != 200:
                        logger.error(f"DuckDuckGo HTML search error: {response.status}")
                        return []
                    
                    html = await response.text()
                    results = []
                    
                    # 검색 결과 파싱
                    # <a class="result__a" href="...">title</a>
                    # <a class="result__snippet">snippet</a>
                    result_blocks = re.findall(
                        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                        html, re.DOTALL
                    )
                    
                    for url, title, snippet in result_blocks[:5]:
                        # HTML 태그 제거
                        title = re.sub(r'<[^>]+>', '', title).strip()
                        snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                        
                        # URL 디코딩
                        if url.startswith('//duckduckgo.com/l/?uddg='):
                            url_match = re.search(r'uddg=([^&]+)', url)
                            if url_match:
                                url = urllib.parse.unquote(url_match.group(1))
                        
                        if title and snippet:
                            results.append({
                                'source': 'duckduckgo',
                                'title': title,
                                'url': url,
                                'content': snippet,
                                'reliability_score': 0.8,
                                'relevance_score': 0.85
                            })
                    
                    # 결과가 없으면 기본 응답
                    if not results:
                        results.append({
                            'source': 'duckduckgo',
                            'title': f'Search: {query}',
                            'url': f'https://duckduckgo.com/?q={urllib.parse.quote(query)}',
                            'content': f'Search results for: {query}',
                            'reliability_score': 0.7,
                            'relevance_score': 0.7
                        })
                    
                    return results
                    
        except asyncio.TimeoutError:
            logger.error("DuckDuckGo HTML search timed out")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo HTML search failed: {str(e)}")
            return []


class FetchAdapter(BaseMCPAdapter):
    """웹 페이지 콘텐츠 가져오기 어댑터"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Fetch는 검색 기능이 없음"""
        return []
    
    async def fetch_url(self, url: str) -> Optional[Dict[str, Any]]:
        """URL에서 콘텐츠 가져오기"""
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        logger.error(f"Fetch error: {response.status} for {url}")
                        return None
                    
                    html = await response.text()
                    
                    # 간단한 HTML 텍스트 추출
                    text = self._extract_text_from_html(html)
                    
                    return {
                        'source': 'web_scraper',
                        'title': self._extract_title(html) or url,
                        'url': url,
                        'content': text[:5000],
                        'reliability_score': 0.7,
                        'relevance_score': 0.8
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"Fetch timed out for {url}")
            return None
        except Exception as e:
            logger.error(f"Fetch failed for {url}: {str(e)}")
            return None
    
    def _extract_text_from_html(self, html: str) -> str:
        """HTML에서 텍스트 추출"""
        import re
        
        # script, style 태그 제거
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # HTML 엔티티 디코딩
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        # 연속 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_title(self, html: str) -> str:
        """HTML에서 제목 추출"""
        import re
        
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        return ''


class MCPClient:
    """통합 MCP 클라이언트 - 모든 MCP 어댑터 관리"""
    
    def __init__(self):
        self.wikipedia = WikipediaAdapter()
        self.arxiv = ArxivAdapter()
        self.duckduckgo = DuckDuckGoAdapter()
        self.fetch = FetchAdapter()
        self._initialized = False
    
    async def initialize(self):
        """초기화"""
        self._initialized = True
        logger.info("MCPClient initialized")
    
    async def close(self):
        """정리"""
        await self.wikipedia.close()
        await self.arxiv.close()
        await self.duckduckgo.close()
        await self.fetch.close()
        self._initialized = False
        logger.info("MCPClient closed")
    
    async def search_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        """위키피디아 검색"""
        return await self.wikipedia.search(query)
    
    async def get_wikipedia_article(self, title: str) -> Optional[Dict[str, Any]]:
        """위키피디아 문서 가져오기"""
        return await self.wikipedia.get_article(title)
    
    async def search_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """ArXiv 논문 검색"""
        return await self.arxiv.search(query)
    
    async def get_arxiv_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """ArXiv 논문 상세 정보"""
        return await self.arxiv.get_paper(arxiv_id)
    
    async def search_web(self, query: str) -> List[Dict[str, Any]]:
        """DuckDuckGo 웹 검색"""
        return await self.duckduckgo.search(query)
    
    async def fetch_url(self, url: str) -> Optional[Dict[str, Any]]:
        """URL 콘텐츠 가져오기"""
        return await self.fetch.fetch_url(url)
    
    async def search_by_type(self, query: str, mcp_type: str) -> List[Dict[str, Any]]:
        """MCP 타입에 따라 적절한 검색 수행"""
        if mcp_type == 'WIKIPEDIA':
            return await self.search_wikipedia(query)
        elif mcp_type == 'ARXIV':
            return await self.search_arxiv(query)
        elif mcp_type in ['DUCKDUCKGO', 'WEB_SEARCH']:
            return await self.search_web(query)
        elif mcp_type == 'WEB_SCRAPER':
            # 웹 검색 후 스크래핑
            search_results = await self.search_web(query)
            scraped_results = []
            for result in search_results[:3]:
                url = result.get('url', '')
                if url and url.startswith('http'):
                    scraped = await self.fetch_url(url)
                    if scraped:
                        scraped_results.append(scraped)
            return scraped_results if scraped_results else search_results
        else:
            return await self.search_web(query)
