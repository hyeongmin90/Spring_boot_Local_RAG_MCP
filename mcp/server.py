import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

load_dotenv(Path(__file__).parent.parent / ".env")

from mcp.server.fastmcp import FastMCP
from data_pipeline.storage import query_hybrid


mcp = FastMCP("spring_docs")

@mcp.tool()
def get_docs(query: str) -> str:
    """
    Get documents for a query.
    spring boot reference documentation
    spring data redis reference documentation
    use english for query
    
    Args:
        query: The search query string.
    Returns:
        List of documents.
    """
    docs = query_hybrid(query, k=5)
    results = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]
    return json.dumps(results, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()