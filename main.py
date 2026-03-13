import os
import sys
import warnings
from dotenv import load_dotenv
from colorama import init, Fore, Style

warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
init(autoreset=True)

from agent import build_graph


BANNER = f"""{Fore.CYAN}
╔══════════════════════════════════════════════════════╗
║         LangGraph RAG Agent  (Spring Docs)           ║
║         'exit' 또는 'q' 를 입력하면 종료됩니다       ║
╚══════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""

CATEGORY_LABELS = {
    "spring-boot": "Spring Boot",
    "spring-data-jpa": "Spring Data JPA",
    "spring-data-redis": "Spring Data Redis",
    "spring-security": "Spring Security",
    "spring-cloud-gateway": "Spring Cloud Gateway",
    None: "전체",
}


def _print_step(label: str, value: str, color: str = Fore.CYAN) -> None:
    print(f"  {color}▶ {label}:{Style.RESET_ALL} {value}")


def run():
    graph = build_graph()
    print(BANNER)

    while True:
        try:
            user_input = input(f"{Fore.GREEN}User:{Style.RESET_ALL} ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break

            print()

            state = {
                "question": user_input,
                "rewritten_query": None,
                "category": None,
                "should_rewrite": False,
                "documents": [],
                "answer": "",
            }

            for step_output in graph.stream(state, stream_mode="updates"):
                node_name = list(step_output.keys())[0]
                node_result = step_output[node_name]

                if node_name == "analyze":
                    should_rw = node_result.get("should_rewrite", False)
                    category = node_result.get("category")
                    cat_label = CATEGORY_LABELS.get(category, str(category))

                    _print_step(
                        "분석",
                        f"카테고리={cat_label}  |  재작성={'필요' if should_rw else '불필요'}",
                        Fore.MAGENTA,
                    )

                elif node_name == "rewrite":
                    rewritten = node_result.get("rewritten_query", "")
                    _print_step("쿼리 재작성", rewritten, Fore.BLUE)

                elif node_name == "retrieve":
                    doc_count = len(node_result.get("documents", []))
                    _print_step("검색 완료", f"{doc_count}개 문서 검색됨", Fore.YELLOW)

                elif node_name == "generate":
                    answer = node_result.get("answer", "")
                    print(f"\n{Fore.YELLOW}Agent:{Style.RESET_ALL}\n{answer}\n")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted. Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error:{Style.RESET_ALL} {e}\n")


if __name__ == "__main__":
    run()
