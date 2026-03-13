from colorama import Fore, Style
from langchain_core.callbacks import BaseCallbackHandler



class PromptInspector(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print(f"\n{Fore.MAGENTA}전체 컨텍스트{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}")
        
        for i, msg in enumerate(messages[0]):
            msg_type = msg.__class__.__name__
            content = msg.content
            
            if msg_type == 'SystemMessage':
                header = f"{Fore.YELLOW}[System Prompt]{Style.RESET_ALL}"
            
            elif msg_type == 'HumanMessage':
                header = f"{Fore.BLUE}[User]{Style.RESET_ALL}"
            
            elif msg_type == 'AIMessage':
                header = f"{Fore.GREEN}[AI Assistant]{Style.RESET_ALL}"
                if msg.tool_calls:
                    content += f"\n{Fore.YELLOW}   └─ 🛠️ Tool Call 요청: {msg.tool_calls}{Style.RESET_ALL}"
            
            elif msg_type == 'ToolMessage':
                tool_name = msg.name if hasattr(msg, 'name') else 'Unknown Tool'
                header = f"{Fore.CYAN}[Tool Result: {tool_name}]{Style.RESET_ALL}"
                if len(content) > 200:
                    content = content[:200] + f"... (+{len(content)-200}자 더 있음)"
            
            else:
                header = f"[{msg_type}]"

            print(f"{i:02d}. {header} : {content}")
            
        print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")