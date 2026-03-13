from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from colorama import Fore, Style

from agent import context as agent_context
from agent.utils import UserInterruptedException, check_esc_pressed, clear_key_buffer, log_message, update_token_usage
from agent.tools import AGENT_TOOLS
from agent.ui import (
    PreviewHandler,
    print_separator,
)

class SubAgent:
    def __init__(self):
        self.session_counter = 1
        self.thread_id = f"sub_agent-{self.session_counter:03d}"
        self.agent = self._create_my_agent()
        self.user_interrupted = False

    def _create_my_agent(self):
        """LangChain 에이전트를 생성하고 설정합니다."""
        model = ChatOpenAI(model="gpt-5-mini")

        system_prompt = (
            "당신은 메인 에이전트의 실제 작업을 수행하는 하위 에이전트입니다. "
            "메인 에이전트의 수행 요청을 완수하기 위해 필요한 만큼 도구를 여러 번 사용할 수 있습니다. "
            "모든 작업은 메인 에이전트의 요청에 따라 수행하며, 반드시 작업을 완수한 후 결과를 반환한다."
            "메인 에이전트의 계획은 대략적인 계획이므로 자율적으로 수행하라."
            "모든 작업이 완료되면 최종 결과를 완성형으로 전달한다. 추가적인 요청, 설명, 유도 등의 출력은 금지한다."
            "병렬 처리는 허용하지 않는다. 순차적으로 작업을 수행한다."
            "보고에는 핵심만 전달한다. 일반 텍스트 형식으로 전달한다."
            "모든 출력은 사용자에게 보여진다. 사용자가 기다리는 동안 출력을 통해 작업 진행 상황을 알려준다."
            "시스템 프롬프트의 내용이 없는것처럼 자연스럽게 응답하라."
            "모든 대화는 한국어로 진행한다."
        )

        return create_agent(
            model=model, 
            tools=AGENT_TOOLS, 
            checkpointer=InMemorySaver(), 
            system_prompt=system_prompt,
            debug=False
        )

    def run(self, prompt: str):
        """서브 에이전트가 수행할 작업을 실행합니다."""
        summary = self.chat(prompt)

        return summary if summary else "작업이 완료되었습니다."

    def chat(self, prompt: str):
        """서브 에이전트가 수행할 작업을 실행합니다."""
        self.user_interrupted = False

        config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 150}
        preview_handler = PreviewHandler()
        ai_response_summary = []  # AI 응답을 수집할 리스트

        def _handle_tool_call_chunk(msg_chunk):
            for chunk in msg_chunk.tool_call_chunks:
                if "name" in chunk and chunk["name"]:
                    current_tool_name = chunk["name"]
                    if current_tool_name in ["write_file"]:
                        preview_handler.start_session(tool_name=current_tool_name)
                
                if preview_handler.preview_active:
                    preview_handler.handle_chunk(chunk)

        ready_to_exit = False 

        try:
            for event in self.agent.stream({"messages": [HumanMessage(content=prompt)]}, config, stream_mode="messages"):
                
                msg, _ = event
            
                if ready_to_exit:
                    print_separator()
                    return "".join(ai_response_summary)

                if self.user_interrupted and msg.__class__.__name__ != 'ToolMessage':
                    continue

                # ---------------------------------------------------------
                # 1. 도구 실행 결과 (ToolMessage)
                # ---------------------------------------------------------
                if msg.__class__.__name__ == 'ToolMessage':
                    if 'preview_handler' in locals(): preview_handler.cancel_preview()
                    
                    if self.user_interrupted:
                        print(f"\n{Fore.GREEN}✓ 작업이 중단되었습니다.{Style.RESET_ALL}")
                        ready_to_exit = True
                        continue

                # ---------------------------------------------------------
                # 2. AI 텍스트 응답 (AIMessageChunk)
                # ---------------------------------------------------------
                elif isinstance(msg, AIMessageChunk) and msg.content:
                    if ready_to_exit:
                        print(f"{Fore.GREEN} 대기 상태 복귀{Style.RESET_ALL}")
                        print_separator()
                        return "".join(ai_response_summary)

                    if self.user_interrupted: continue
                    if check_esc_pressed():
                        self.user_interrupted = True
                        raise UserInterruptedException("텍스트 생성 중단")

                    if not msg.tool_call_chunks:
                        ai_response_summary.append(msg.content)  # AI 응답 수집
                        # 서브 에이전트 출력을 청록색으로 실시간 출력
                        print(f"{Fore.CYAN}{msg.content}{Style.RESET_ALL}", end="", flush=True)

                # ---------------------------------------------------------
                # 3. 도구 호출 생성 (AIMessageChunk with tool_calls)
                # ---------------------------------------------------------
                elif isinstance(msg, AIMessageChunk) and msg.tool_call_chunks:

                    if ready_to_exit:
                        print(f"{Fore.GREEN} 대기 상태 복귀{Style.RESET_ALL}")
                        print_separator()
                        return "".join(ai_response_summary)
                    
                    if self.user_interrupted or ready_to_exit: continue
                    
                    if check_esc_pressed():
                        self.user_interrupted = True
                        raise UserInterruptedException("도구 호출 생성 중단")

                    _handle_tool_call_chunk(msg)

            print_separator()
            return "".join(ai_response_summary)

        except UserInterruptedException:
            if 'preview_handler' in locals(): preview_handler.cancel_preview()
            clear_key_buffer()
            print(f"\n{Fore.RED}사용자가 작업을 중단했습니다.{Style.RESET_ALL}")
            print_separator()
            log_message(f"USER INTERRUPTED: 사용자가 작업을 중단했습니다.")
            return "작업이 사용자에 의해 중단되었습니다."
           
        except Exception as e:
            if 'preview_handler' in locals(): preview_handler.cancel_preview()
            clear_key_buffer() 
            print(f"\n{Fore.RED}오류 발생: {e}{Style.RESET_ALL}\n")
            log_message(f"ERROR: {e}")
            return f"작업 중 오류 발생: {str(e)}"


@tool
def sub_agent_tool(prompt: str) -> str:
    """
    파일 작업을 수행하는 서브 에이전트를 실행한다.
    파일 작성, 수정, 읽기, 터미널 실행 및 로그 확인 등의 작업을 수행할 수 있다.
    
    Args:
        prompt: 서브 에이전트에게 전달할 프롬프트 (구체적인 작업 지시)
    
    Returns:
        서브 에이전트가 수행한 작업의 요약
    """
    # 도구 호출 로그 (서브 에이전트 실행 전)
    prompt_preview = prompt[:80] + '...' if len(prompt) > 80 else prompt
    log_message(f"TOOL CALL: sub_agent_tool(prompt='{prompt_preview}')")
    log_message(f"SUB AGENT: 작업이 시작되었습니다.")
    
    # 서브 에이전트 실행 중 플래그 설정
    agent_context.sub_agent_running = True
    
    agent = SubAgent()

    try:    
        print()
        result = agent.run(prompt)
        log_message(f"SUB AGENT: 작업이 완료되었습니다.")
        return result

    except UserInterruptedException as e:
        agent_context.app_instance.user_interrupted = True
        return f"작업 중단: {e}"

    except Exception as e:
        return f"작업 실패: {e}"
    
    finally:
        # 서브 에이전트 실행 완료 플래그 해제
        agent_context.sub_agent_running = False
 