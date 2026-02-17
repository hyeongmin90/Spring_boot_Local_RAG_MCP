import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def get_content(url):
    """페이지 본문 추출"""
    try:
        resp = requests.get(url, headers=headers, timeout=300)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for tag in soup(["nav", "footer", "header", "script", "style", "aside"]):
            tag.decompose()
        
        content = soup.find("article", class_="doc") or soup.find("main") or soup.find("div", {"id": "content"})
        
        if not content:
            return ""

        text = content.get_text()
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text
        
    except Exception as e:
        print(f"    Error: {e}")
        return ""
    return ""


def extract_path_from_url(url):
    """
    URL에서 경로 추출
    """

    if '/spring-boot/' in url:
        path = url.split('/spring-boot/')[-1]
        # .html 제거
        path = path.replace('.html', '')
        # 슬래시를 언더스코어로
        path = path.replace('/', '_')
        return path
    else:
        return url.split('/')[-1].replace('.html', '')

def fetch_spring_boot_docs(start_url="https://docs.spring.io/spring-boot/reference/", max_pages=None):
    """Spring Boot 문서 크롤링"""
    
    print(f"Fetching index from {start_url}...")
    
    try:
        response = requests.get(start_url, headers=headers, timeout=30)
        response.raise_for_status()
        index_html = response.text
    except requests.RequestException as e:
        print(f"Error: {e}")
        return
    
    soup = BeautifulSoup(index_html, 'html.parser')
    
    base_domain = urlparse(start_url).netloc
    base_path = '/spring-boot/reference'
    
    # 링크 수집
    links_to_visit = set()
    
    print("Scanning for links...")
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(start_url, href)
        
        # 앵커 제거
        clean_url = full_url.split('#')[0]
        
        parsed_url = urlparse(clean_url)
        
        # 필터링: 같은 도메인, html 경로 내, html 파일
        if (parsed_url.netloc == base_domain and 
            base_path in parsed_url.path and
            clean_url.endswith('.html') and
            clean_url != start_url):
            links_to_visit.add(clean_url)
    
    print(f"Found {len(links_to_visit)} unique URLs")
    
    # 출력 디렉토리
    output_dir = "spring_boot_crawled"
    os.makedirs(output_dir, exist_ok=True)
    
    links_list = list(links_to_visit)
    
    if max_pages:
        print(f"Limiting to {max_pages} pages")
        links_list = links_list[:max_pages]
    
    print(f"\nCrawling {len(links_list)} pages...")
    print("="*80)
    
    success_count = 0
    saved_files = {}
    
    for i, url in enumerate(links_list, 1):
        # 경로 기반 파일명 생성
        filepath_name = extract_path_from_url(url)
        
        print(f"[{i}/{len(links_list)}] {filepath_name}", flush=True)
        print(f"  URL: {url}")
        
        content = get_content(url)
        
        if content and len(content) > 200:
            # 파일명 정리
            filename = filepath_name
            if not filename:
                filename = f"page_{i}"
            
            # 중복 처리
            if filename in saved_files:
                counter = 1
                base = filename
                while f"{base}_{counter}" in saved_files:
                    counter += 1
                filename = f"{base}_{counter}"
            
            saved_files[filename] = url
            filename += ".txt"
            
            # 저장
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"{content}")
            
            print(f"  ✓ Saved: {filename} ({len(content):,} chars)")
            success_count += 1
            
            yield {'url': url, 'content': content}
        else:
            print(f"  ✗ Content too short or empty")
        
        time.sleep(0.1)  # Rate limiting
    
    print("\n" + "="*80)
    print(f"Completed: {success_count}/{len(links_list)} pages saved")
    print(f"Output directory: {output_dir}/")
    
    # 통계
    import glob
    saved = glob.glob(os.path.join(output_dir, "*.txt"))
    total_size = sum(os.path.getsize(f) for f in saved)
    
    print(f"\nStatistics:")
    print(f"  Total files: {len(saved)}")
    print(f"  Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

if __name__ == "__main__":
    for _ in fetch_spring_boot_docs(max_pages=5):
        pass