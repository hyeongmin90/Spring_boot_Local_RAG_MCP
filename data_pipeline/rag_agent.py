import sys
import os
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from colorama import init, Fore, Style

from data_pipeline.storage import query_documents

# Initialize Colorama
init(autoreset=True)

# Global set to track seen chunk IDs per turn
SEEN_IDS = set()

# Define the Tool
@tool
def search_spring_boot_docs(query: str, category: str = None) -> str:
    """
    Searches the Spring Boot reference documentation for relevant information.
    Use this tool to find answers to questions about Spring Boot configuration, features, and usage.
    
    Docs Version Info: Spring Boot 4.0.2, Spring Data Redis 4.0.3

    Args:
        query: The search query string.
        category: The type of documentation to search. (defalut: None, Type = [spring-boot, spring-data-redis])
    """
    global SEEN_IDS
    try:
        # Fetch more results to allow for filtering
        raw_results = query_documents(query, 5, category)

        print(f"Searching {query[:20]}...\n")
        
        results = []
        for doc in raw_results:
            # Use chunk_id if available, otherwise just skip if we can't uniquely identify
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                # Fallback: allow if no chunk_id (legacy support)
                results.append(doc)
                continue
                
            if chunk_id in SEEN_IDS:
                continue
                
            SEEN_IDS.add(chunk_id)
            results.append(doc)
            
            if len(results) >= 5:
                break
        
        if not results:
            return "No new results found in the documentation."
        
        with open("RagAgent_SearchLog.txt", "a", encoding="utf-8") as f:
            f.write(f"query: {query}\n")
            f.write(f"category: {category}\n")
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "Unknown")
                header = doc.metadata.get("header", "N/A")
                f.write(f"Result {i}" + '-' * 50 + "\n")
                f.write(f"Source: {source}\n")
                f.write(f"Header: {header}\n")
                f.write(f"Category: {doc.metadata.get('category', 'Unknown')}\n")
                f.write(f"Content:\n{doc.page_content}\n")
            f.write("="*50 + "\n")

        output = ""
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            header = doc.metadata.get("header", "N/A")

            output += f"--- Result {i} ---\n"
            output += f"Source: {source}\n"
            output += f"Header: {header}\n"
            output += f"Category: {doc.metadata.get('category', 'Unknown')}\n"
            output += f"Content:\n{doc.page_content}\n" 
            output += "\n"

        return output
    except Exception as e:
        return f"Error during search: {e}"

def run_rag_agent():
    load_dotenv()
    
    # Initialize Model
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    # Define Tools
    tools = [search_spring_boot_docs]
    
    # System Prompt (String)
    system_prompt = (
        "You are a Spring Boot Expert RAG Agent.\n"
        "Answer user questions accurately using the provided documentation.\n"
        "ALWAYS use the 'search_spring_boot_docs' tool to verify information before answering.\n"
        "Do not specify a document type for the initial search. Specify the document type for subsequent searches.\n"
        "If you cannot find the answer in the search results, admit it honestly.\n"
        "Do not include any information not found in the search results.\n"
        "If search results are insufficient, retry with different keywords before giving up.\n"
        "Provide clear, code-centric answers where applicable.\n"
        "Answer in Korean."
    )
    
    # Create Agent (Using user's custom/specific create_agent function if available in env)
    # The user insists on 'create_agent' which is likely imported from langchain.agents in their setup.
    try:
        agent = create_agent(
            model=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            system_prompt=system_prompt,
            debug=False
        )
    except ImportError:
        print("Error: 'create_agent' not found in langchain.agents. Please check your environment.")
        return

    print(f"{Fore.CYAN}=== Spring Boot RAG Agent (Type 'exit' to quit) ==={Style.RESET_ALL}")
    
    thread_id = "rag-cli-session"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            user_input = input(f"\n{Fore.GREEN}User: {Style.RESET_ALL}").strip()

            with open("RagAgent_SearchLog.txt", "a", encoding="utf-8") as f:
                f.write(f"User: {user_input}\n")
                f.write("="*50 + "\n")

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
            
            # Reset seen chunks for new user query
            SEEN_IDS.clear()
            
            print(f"\n{Fore.YELLOW}Agent:{Style.RESET_ALL} ", end="", flush=True)
            
            full_response = ""

            # Stream execution
            for event in agent.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="messages"):
                msg, _ = event
                
                # Print Text Content
                if isinstance(msg, AIMessageChunk) and msg.content:
                    # Skip tool calls if they are just chunks of arguments
                    if not msg.tool_call_chunks: 
                        print(msg.content, end="", flush=True)
                        full_response += msg.content

                # Optional: Indicate Tool Usage
                # if msg.__class__.__name__ == 'ToolMessage':
                #    print(f"\n[Tool Result] {msg.content[:50]}...", end="")

            print() # Newline after response

            with open("RagAgent_SearchLog.txt", "a", encoding="utf-8") as f:
                f.write(f"\nAgent: {full_response}\n")
                f.write("="*50 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    run_rag_agent()
