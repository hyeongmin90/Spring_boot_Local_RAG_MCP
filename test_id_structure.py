import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_pipeline.storage import query_documents
from dotenv import load_dotenv

load_dotenv()

docs = query_documents("Spring Boot", k=1)
if docs:
    doc = docs[0]
    print(f"doc attribute 'id': {getattr(doc, 'id', 'NOT FOUND')}")
    print(f"doc.metadata keys: {doc.metadata.keys()}")
    print(f"doc.metadata: {doc.metadata}")
else:
    print("No documents found.")
