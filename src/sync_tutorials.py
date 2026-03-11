import sys
import os
import time
import csv
import argparse
import pygame
import visualization_utils as vu
from main import main as run_overcooked_game, get_parser
from tutorial import run_tutorial

def log_summary_score(pid, block, layout, cond_id, cond_name, score, timestep, inter_col, duration):
    log_dir = f"experiments/PID_{pid}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    file_path = os.path.join(log_dir, f"summary_PID_{pid}.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["PID", "Block", "Map_Key", "Map_Name", "Condition_ID", "Condition_Name", "Score", "Timestep", "Inter_Collisions", "Duration_Sec", "Timestamp"])
        writer.writerow([pid, block, layout[0], layout[1], cond_id, cond_name, score, timestep, inter_col, f"{duration:.2f}", time.strftime("%Y-%m-%d %H:%M:%S")])

def run_tutorial_step(pid, multi, is_async, visual_level, layout, title, description, log_name, step_id, screen):
    # 1. 시작 대기 화면 (U 키 대기) - 원래대로 롤백
    vu.draw_centered_text(screen, f"[멀티 비실시간 튜토리얼 {step_id}/2] {title}", f"{description} | 'U' 키를 눌러 시작하세요", color=(0, 255, 100))
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_u: waiting = False
    
    # 2. 로딩 화면 표시
    vu.draw_centered_text(screen, "잠시만 기다려주세요...", "튜토리얼 환경 및 AI를 초기화 중입니다.", color=(255, 200, 0))
    pygame.display.flip()

    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
    from utils import make_agent

    class Args: pass
    args = Args()
    args.is_async, args.visual_level, args.delay, args.log_dir, args.name = is_async, visual_level, 400, f"experiments/PID_{pid}/logs", log_name

    # 게임 중에 띄울 설명 텍스트를 args에 추가하여 tutorial.py 내부에서 사용할 수 있게 전달
    if visual_level == 2:
        args.guide_text = "초록색: 에이전트의 plan / 파란색: 에이전트가 추론한 player의 plan (본인의 plan만 띄울 수도 있고, plan+추론 과정일 수도 있습니다.)"
    elif visual_level == 1:
        args.guide_text = "상단: 추론한 player의 plan / 하단: 에이전트의 plan (본인의 plan만 띄울 수도 있고, plan+추론 과정일 수도 있습니다.)"
    else:
        args.guide_text = ""

    mdp = OvercookedGridworld.from_grid(["XXXXPXXXX", "O       D", "X      2X", "X   1   X", "XXXXSXXXX"])
    env = OvercookedEnv(mdp=mdp, horizon=9999)
    vis = StateVisualizer()
    
    ai = None
    if multi:
        ai = make_agent('EIRAAsync', mdp, "tutorial_grid", K=1)
        # AttributeError 방지 속성 강제 주입
        ai.set_agent_index(1)
        ai.async_mode = is_async
        ai.teammate_intentions_dict = {}
        ai.current_timestep = 0
        ai.reset()
    
    start_t = time.time()
    score, step, col = run_tutorial(env, screen, vis, ai_agent=ai, args=args)
    log_summary_score(pid, 0, ("Phase1", "tutorial_grid"), -100 - step_id, log_name, score, step, col, time.time() - start_t)

def run_main_step(pid, layout, is_async, visual_level, cond_name, title, description, screen):
    # 1. 시작 대기 화면 (U 키 대기) - 원래대로 롤백
    vu.draw_centered_text(screen, f"[적응 단계] ", f" 'U' 키를 눌러 시작하세요", color=(0, 255, 100))
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_u: waiting = False
            
    # 2. 로딩 화면 표시
    vu.draw_centered_text(screen, "잠시만 기다려주세요...", "본 실험 환경 및 AI를 초기화 중입니다.", color=(255, 200, 0))
    pygame.display.flip()

    # 게임 중에 띄울 설명 텍스트 설정
    guide_text = ""
    if visual_level == 2:
        guide_text = "초록색: 에이전트의 plan / 파란색: 에이전트가 추론한 player의 plan (본인의 plan만 띄울 수도 있고, plan+추론 과정일 수도 있습니다.)"
    elif visual_level == 1:
        guide_text = "상단: 추론한 player의 plan / 하단: 에이전트의 plan (본인의 plan만 띄울 수도 있고, plan+추론 과정일 수도 있습니다.)"

    start_t = time.time()
    
    # config 딕셔너리에 guide_text를 추가하여 main.py로 전달
    score, step, col = run_overcooked_game({
        'layout': layout, 'async': is_async, 'visual_level': visual_level, 'name': cond_name, 
        'log_dir': f"experiments/PID_{pid}/logs", 'episode': 1, 'horizon': 400, 
        'p0': 'Human', 'p1': 'EIRAAsync', 'show_intention': True, 'render': True,
        'guide_text': guide_text
    }, surface=screen)
    log_summary_score(pid, 0, ("Phase1", layout), -1 if "Baseline" in cond_name else -2, cond_name, score, step, col, time.time() - start_t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args(); pid = args.pid
    pygame.init(); screen = pygame.display.set_mode((900, 750))
    pygame.display.set_caption(f"Phase 1 - PID: {pid}")
    
    tut_steps = [
        # (False, False, 0, "비실시간", "", "Tut_S_NRT"), 
        # (False, True, 0, "실시간", "", "Tut_S_RT"), 
        # (True, False, 2, "", "", "Tut_M_NRT_H"), 
        # (True, False, 1, "", "", "Tut_M_NRT_N"), 
        (True, False, 2, "", "하이라이트", "Tut_M_RT_H"), 
        (True, False, 1, "", "자연어", "Tut_M_RT_N")
    ]
    
    for i, (m, a, v, title, desc, ln) in enumerate(tut_steps):
        # "tutorial_grid" 인자를 추가하여 총 10개의 인자를 전달합니다.
        run_tutorial_step(pid, m, a, v, "tutorial_grid", title, desc, ln, i+1, screen)
    
    # run_main_step(pid, "simple", True, 2, "Phase1_Practice", "연습 세션", "본 맵에서 AI 하이라이트를 보며 연습합니다", screen)
    run_main_step(pid, "cramped_room", False, 0, "Phase1_Baseline", "베이스라인 측정", "아무 정보 없이 협업을 수행합니다", screen)
    
    pygame.quit()