import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def get_content_as_markdown(url: str) -> str:
    """
    Fetches the HTML content of the target URL and accurately converts the core 
    article/document area into Markdown format, stripping out navigation/footers.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        resp.encoding = 'utf-8'
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove noisy tags
        for tag in soup(["nav", "footer", "header", "script", "style", "aside"]):
            tag.decompose()
        
        # Locate core content
        content = soup.find("article", class_="doc") or soup.find("main") or soup.find("div", {"id": "content"})
        
        if not content:
            return ""

        # Convert to Markdown (ATX style headers, strip a tags to avoid link clutter)
        html_content = str(content)
        text = md(html_content, heading_style="ATX", strip=['a'])
        
        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()
        
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return ""
