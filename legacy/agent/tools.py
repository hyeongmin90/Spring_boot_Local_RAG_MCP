
import subprocess
import platform
import time
import shutil
import os
import sys
from pathlib import Path

from langchain_core.tools import tool
from colorama import Fore, Style
from . import context
from .context import approval_lock, BASE_DIR, CODE_DIR
from .utils import is_safe_path, check_esc_pressed, UserInterruptedException, clear_key_buffer
from .ui import get_separator_line, wrap_text_wide, TerminalOutputViewer

project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)

from final_pipeline.agent import ask_query

# ==========================================
# 도구(Tools) 정의 및 관련 헬퍼
# ==========================================


def _request_approval(prompt: str) -> bool:
    """사용자에게 작업을 승인받는 중앙 함수"""
    app = context.app_instance
    if app and app.auto_approve_mode:
        print(f"{Fore.GREEN}[자동 승인]\n {prompt}\n{Style.RESET_ALL}")
        return True
    
    with approval_lock:
        print(f"\n{get_separator_line(color=Fore.YELLOW)}")
        print(f"\n{prompt}")
        print(f"\n{get_separator_line(color=Fore.YELLOW)}")
        approval = input(f"\n{Fore.YELLOW}실행하시겠습니까? (y/n): {Style.RESET_ALL}").strip().lower()
        print(f"\n{get_separator_line(color=Fore.YELLOW)}")
        return approval == 'y'

def _build_tree(directory: Path, prefix: str = "", max_depth: int = 6, current_depth: int = 0) -> tuple[str, int]:
    if current_depth >= max_depth:
        return f"{prefix}└─ ... \n", 1
    
    try:
        excluded_names = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vscode'}
        items = [item for item in sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                 if item.name not in excluded_names and not item.name.startswith('.')]
    except PermissionError:
        return f"{prefix}└─ ... (접근 권한 없음)\n", 1
    
    tree_lines, item_count = [], 0
    max_files_in_subdir = 4
    is_root = (current_depth == 0)
    
    files = [item for item in items if item.is_file()]
    dirs = [item for item in items if item.is_dir()]
    
    for i, item in enumerate(dirs):
        is_last_dir = (i == len(dirs) - 1) and len(files) == 0
        connector = "└─ " if is_last_dir else "├─ "
        tree_lines.append(f"{prefix}{connector} {item.name}/\n")
        item_count += 1
        
        extension = "   " if is_last_dir else "│  "
        subtree, sub_count = _build_tree(item, prefix + extension, max_depth, current_depth + 1)
        tree_lines.append(subtree)
        item_count += sub_count
    
    files_to_show = files if is_root else files[:max_files_in_subdir]
    
    for i, item in enumerate(files_to_show):
        is_last = (i == len(files_to_show) - 1)
        connector = "└─ " if is_last else "├─ "
        tree_lines.append(f"{prefix}{connector} {item.name}\n")
        item_count += 1
    
    if not is_root and len(files) > max_files_in_subdir:
        remaining = len(files) - max_files_in_subdir
        tree_lines.append(f"{prefix}└─ ... {remaining}개 파일\n")
        item_count += 1
    
    return "".join(tree_lines), item_count

@tool
def list_files(path: str = ".", max_depth: int = 1) -> str:
    """
    파일 목록을 트리 형태로 조회합니다.
    max_depth: 최대 깊이 (기본값: 1)
    """

    if not is_safe_path(path, BASE_DIR):
        return "보안 경고: 작업 디렉토리 외부 경로는 조회할 수 없습니다."
    try:
        target = (BASE_DIR / path).resolve()
        if not target.exists(): return "경로가 존재하지 않습니다."
        if not target.is_dir(): return "디렉터리가 아닙니다."
        
        tree, count = _build_tree(target, "", max_depth)
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        print(f"\n{Style.BRIGHT}ReadFileList{Style.RESET_ALL} {path}")
        print(f"Found {count} items")
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        
        return f"{target}\n{tree}\n총 {count}개 항목"
    except Exception as e:
        return f"오류: {e}"

@tool
def read_file(filename: str) -> str:
    """파일 내용을 읽습니다."""
    if not is_safe_path(filename, BASE_DIR):
        return "보안 경고: 작업 디렉토리 외부 파일은 읽을 수 없습니다."
    try:
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        print(f"\n{Style.BRIGHT}FileRead{Style.RESET_ALL} {filename}")
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")

        target = (BASE_DIR / filename).resolve()
        if not target.is_file(): 
            return "파일이 아니거나 존재하지 않습니다."
        return f"내용:\n{target.read_text(encoding='utf-8')}"
    except Exception as e:
        return f"읽기 오류: {e}"

@tool
def write_file(filename: str, content: str) -> str:
    """파일을 새로 쓰거나 덮어씁니다. 경로는 자동으로 생성된다."""
    if not is_safe_path(filename, BASE_DIR):
        return "보안 경고: 작업 디렉토리 외부에 파일을 쓸 수 없습니다."
    target = (BASE_DIR / filename).resolve()
    if not _request_approval(f"Write File: {str(target)}"):
        return "사용자가 파일 쓰기를 거부했습니다."
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return f"저장 완료: {str(target)}"
    except Exception as e:
        return f"쓰기 오류: {e}"

def format_diff_with_lines(start_line: int, old_text: str, new_text: str) -> str:
    try:
        cols, _ = shutil.get_terminal_size()
    except:
        cols = 80
    
    display_width = max(cols - 10, 40) 
    result = []

    old_lines = old_text.split('\n')
    for i, line in enumerate(old_lines):
        current_line_num = start_line + i
        if line:
            wrapped = wrap_text_wide(line, display_width)
            for w_idx, w_line in enumerate(wrapped):
                num_str = str(current_line_num) if w_idx == 0 else "."
                result.append(f"{num_str:>4}│ {Fore.RED}- {w_line}{Style.RESET_ALL}")
        else:
            result.append(f"{current_line_num:>4}│ {Fore.RED}- {Style.RESET_ALL}")

    result.append(get_separator_line(char='─', color=Fore.WHITE))

    new_lines = new_text.split('\n')

    for i, line in enumerate(new_lines):
        current_line_num = start_line + i
        if line:
            wrapped = wrap_text_wide(line, display_width)
            for w_idx, w_line in enumerate(wrapped):
                num_str = str(current_line_num) if w_idx == 0 else "."
                result.append(f"{num_str:>4}│ {Fore.GREEN}+ {w_line}{Style.RESET_ALL}")
        else:
            result.append(f"{current_line_num:>4}│ {Fore.GREEN}+{Style.RESET_ALL}")

    return "\n".join(result)

@tool
def edit_file(filename: str, target_text: str, replacement_text: str) -> str:
    """파일의 특정 부분을 수정합니다."""
    if not is_safe_path(filename, BASE_DIR):
        return "보안 경고: 작업 디렉토리 외부 파일은 수정할 수 없습니다."
    
    target_path = (BASE_DIR / filename).resolve()
    if not target_path.exists(): return "오류: 파일이 존재하지 않습니다."

    try:
        content = target_path.read_text(encoding='utf-8')
        
        count = content.count(target_text)
        if count == 0:
            return "오류: 파일에서 target_text를 찾을 수 없습니다."
        elif count > 1:
            return f"오류: target_text가 {count}번 발견되었습니다. 더 긴 문맥을 사용하여 유일한 부분을 지정해주세요."

        start_index = content.find(target_text)
        start_line = content[:start_index].count('\n') + 1

        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        print(f"\n{Style.BRIGHT}Edit File{Style.RESET_ALL} {filename}")
        
        prompt_content = format_diff_with_lines(start_line, target_text, replacement_text)

        if not _request_approval(prompt_content):
            return "사용자가 파일 수정을 거부했습니다."
        
        new_content = content.replace(target_text, replacement_text)
        target_path.write_text(new_content, encoding='utf-8')
        
        return f"수정 완료: {filename} ({start_line}번째 줄 수정됨)"

    except Exception as e:
        return f"수정 오류: {e}"
    finally:
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")


def _decode_bytes_output(output_bytes: bytes) -> str:
    if not output_bytes: return ""
    for enc in ("utf-8", "cp949", "latin-1"):
        try: return output_bytes.decode(enc)
        except UnicodeDecodeError: continue
    return str(output_bytes)

def _sanitize_command_for_filename(command: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    sanitized = command
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    sanitized = sanitized.replace(' ', '_')
    
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    
    sanitized = sanitized.strip('_')
    
    timestamp = int(time.time())
    return f"{sanitized}_{timestamp}.log"

@tool
def run_terminal_command(command: str) -> str:
    """
    터미널 명령어를 실행한다.
    최대 10초의 실행 로그를 반환하며, 초과시 백그라운드로 전환된다.
    백그라운드 로그는 view_terminal_log 도구를 사용하여 확인할 수 있다.


    Args:
        command: 실행할 터미널 명령어
    """

    danger_patterns = ["rm -rf /", "rm -rf", "sudo", "mkfs", ":(){ :|:& };:"]
    if any(pat in command.lower() for pat in danger_patterns):
        return "보안 경고: 위험한 명령어 패턴이 감지되어 실행이 차단되었습니다."
        
    print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
    print(f"\n{Style.BRIGHT}RunTerminalCommand{Style.RESET_ALL}")
    print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")

    if not _request_approval(f"Run: {command}"):
        return "사용자가 명령 실행을 거부했습니다."

    app = context.app_instance

    log_dir = CODE_DIR / "temp_logs"
    log_dir.mkdir(exist_ok=True)
    
    log_name = _sanitize_command_for_filename(command)
    log_path = log_dir / log_name
    
    try:
        if platform.system() == "Windows" and not command.lower().startswith("powershell"):
            final_command = f"chcp 65001 >nul & {command}"
        else:
            final_command = command
        
        with open(str(log_path), "wb") as log_file:
            process = subprocess.Popen(
                final_command, 
                shell=True, 
                cwd=str(BASE_DIR), 
                stdout=log_file, 
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL
            )

        viewer = TerminalOutputViewer(str(log_path), max_lines=10)
        viewer.start(command)
        
        start_time = time.time()
        detached = False

        max_display_time = 10.0
        
        while process.poll() is None:
            viewer.update()
            
            if check_esc_pressed():
                # 프로세스 트리 전체 종료 (Windows: taskkill, Unix: terminate)
                pid = process.pid
                if platform.system() == "Windows":
                    try:
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], 
                                     capture_output=True, timeout=2)
                    except:
                        process.kill()
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                
                viewer.stop(f"{Fore.RED}사용자가 중단했습니다.{Style.RESET_ALL}")
                clear_key_buffer()
                
                # 프로세스 종료 대기
                time.sleep(0.3)
                
                try:
                    log_path.unlink()
                except:
                    pass
                
                if app:
                    app.user_interrupted = True
                
                raise UserInterruptedException("사용자가 명령어 실행을 중단했습니다.")
            
            if time.time() - start_time > max_display_time:
                detached = True
                break
            
            time.sleep(0.1)
        
        if detached:
            viewer.stop(f"{Fore.CYAN} 백그라운드 전환(PID: {process.pid}){Style.RESET_ALL}")

            try:
                with open(str(log_path), 'rb') as f:
                    partial_output = _decode_bytes_output(f.read())
            except:
                partial_output = "(출력을 읽을 수 없습니다)"
            
            if app:
                app.background_processes.append({
                    'process': process,
                    'log_file': log_name,
                    'pid': process.pid,
                    'command': command[:50],
                    'start_time': time.time()
                })
            
            return f"{partial_output}\n...\n 백그라운드 전환 (PID: {process.pid})\n 로그 파일: {log_name}"
        
        else:
            time.sleep(0.1)
            viewer.update()
            
            return_code = process.returncode
            status_color = Fore.GREEN if return_code == 0 else Fore.RED
            
            viewer.stop(f"{status_color} 실행 완료 (종료코드: {return_code}){Style.RESET_ALL}")
            
            try:
                with open(str(log_path), 'rb') as f:
                    output = _decode_bytes_output(f.read())
            except:
                output = "(출력을 읽을 수 없습니다)"
            
            try:
                log_path.unlink()
            except:
                pass
            
            if len(output) > 2000:
                summary = f"...(생략)...\n{output[-2000:]}"
            else:
                summary = output if output else "(출력 없음)"
            
            return summary

    except UserInterruptedException as e:
        if app:
            app.user_interrupted = True
        return "[사용자 중단] 명령어 실행이 취소되었습니다."
    except Exception as e:
        try:
            if 'log_path' in locals():
                log_path.unlink()
        except:
            pass
        return f"실행 실패: {e}"

@tool
def list_background_processes() -> str:
    """백그라운드로 전환된 명령어들의 상태를 확인합니다."""
    app = context.app_instance
    if not app or not app.background_processes:
        return "백그라운드 프로세스가 없습니다."

    result = "백그라운드 프로세스 목록\n"
    
    for i, bg_info in enumerate(app.background_processes, 1):
        proc = bg_info['process']
        is_running = proc.poll() is None
        status = "실행 중" if is_running else f"종료됨 (코드: {proc.returncode})"
        
        elapsed = int(time.time() - bg_info['start_time'])
        elapsed_str = f"{elapsed//60}분 {elapsed%60}초" if elapsed >= 60 else f"{elapsed}초"
        
        result += f"{i}. PID: {bg_info['pid']} - {status}\n"
        result += f"   명령어: {bg_info['command']}\n"
        result += f"   로그: {bg_info['log_file']}\n"
        result += f"   실행 시간: {elapsed_str}\n"
    
    return result

@tool
def view_terminal_log(log_file: str, lines: int = 50) -> str:
    """
    터미널 명령어의 로그를 확인합니다.
    
    Args:
        log_file: 로그 파일명 (list_background_processes에서 확인 가능)
        lines: 마지막 N줄만 표시 (기본값: 50)
    """
    log_dir = CODE_DIR / "temp_logs"
    log_path = log_dir / log_file
    
    try:
        if not log_path.exists():
            return f"로그 파일이 존재하지 않습니다: {log_file}"
        
        with open(log_path, 'rb') as f:
            content = _decode_bytes_output(f.read())
        
        all_lines = content.split('\n')
        
        if len(all_lines) > lines:
            selected_lines = all_lines[-lines:]
            content = f"...(총 {len(all_lines)}줄 중 마지막 {lines}줄)\n" + "\n".join(selected_lines)
        else:
            content = "\n".join(all_lines)
        
        if not content.strip():
            content = "(출력 없음)"

        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        print(f"\n{Style.BRIGHT}ViewTerminalLog{Style.RESET_ALL} {log_file}")
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        
        return content
        
    except Exception as e:
        return f"로그 읽기 실패: {e}"

@tool
def kill_background_process(pid: int) -> str:
    """
    백그라운드 프로세스를 강제 종료합니다.
    
    Args:
        pid: 종료할 프로세스의 PID (list_background_processes에서 확인 가능)
    """
    app = context.app_instance
    if not app or not app.background_processes:
        return "백그라운드 프로세스가 없습니다."
    
    # 해당 PID 찾기
    bg_info = None
    for info in app.background_processes:
        if info['pid'] == pid:
            bg_info = info
            break
    
    if not bg_info:
        return f"PID {pid}를 찾을 수 없습니다. list_background_processes로 확인하세요."
    
    process = bg_info['process']
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                app.background_processes.remove(bg_info)
                return f"프로세스 {pid} ({bg_info['command']})가 종료되었습니다."
            else:
                return f"프로세스 종료 실패 (PID: {pid}): {result.stderr.decode('cp949', errors='ignore')}"
        else:
            process.terminate()
            process.wait(timeout=2)
            app.background_processes.remove(bg_info)
            return f"프로세스 {pid} ({bg_info['command']})가 종료되었습니다."
    except subprocess.TimeoutExpired:
        process.kill()
        app.background_processes.remove(bg_info)
        return f"프로세스 {pid} ({bg_info['command']})가 강제 종료되었습니다."
    except Exception as e:
        return f"프로세스 종료 실패 (PID: {pid}): {e}"
    
@tool
def stand_by() -> str:
    """백그라운드 명령어 실행 로그확인을 위해 10초 동안 대기합니다."""
    
    seconds = 10

    print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
    print(f"\n{Style.BRIGHT}AgentStandByMode{Style.RESET_ALL} {seconds} seconds...")
    print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")

    time.sleep(seconds)
    return f"{seconds}초 대기완료"

@tool
def search_docs(query: str, category: str = None) -> str:
    """
    RAG 파이프라인(공식 문서 데이터베이스)에 질문을 검색하고 답변을 반환합니다.
    최신 프레임워크나 라이브러리 사용법, 예제 코드 등에 대한 정보가 필요할 때 이 도구를 사용하세요.
    
    spring-boot, spring-data-jpa, spring-data-redis, spring-cloud-gateway, spring-security
    
    Args:
        query: 검색할 질문 내용 
        category: (선택) 검색할 카테고리 (예: spring-boot)
    """
    print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
    print(f"\n{Style.BRIGHT}SearchDocs (RAG){Style.RESET_ALL} query: {query}")
    print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
    
    try:
        return ask_query(query, category)
    except Exception as e:
        return f"문서 검색 실패: {e}"

AGENT_TOOLS = [list_files, read_file, write_file, edit_file, run_terminal_command, list_background_processes, view_terminal_log, kill_background_process, stand_by, search_docs]
