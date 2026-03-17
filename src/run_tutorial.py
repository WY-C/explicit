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
    # 1. 시작 대기 화면 (U 키 대기)
    desc_text = f"{description} | " if description else ""
    vu.draw_centered_text(screen, f"[Tutorial {step_id}/2] {title}", f"{desc_text}Press 'U' to start", color=(0, 255, 100))
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_u: waiting = False
        # 무한 루프 과부하 방지를 위한 10ms 대기
        pygame.time.delay(10)
    
    # 2. 로딩 화면 표시
    vu.draw_centered_text(screen, "Please wait...", "Initializing tutorial environment and AI...", color=(255, 200, 0))
    pygame.display.flip()

    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
    from utils import make_agent

    class Args: pass
    args = Args()
    args.is_async = is_async
    args.visual_level = visual_level
    args.delay = 200
    args.log_dir = f"experiments/PID_{pid}/logs"
    args.name = log_name
    args.target_score = 40 

    # 게임 중에 띄울 설명 텍스트를 args에 추가하여 tutorial.py 내부에서 사용할 수 있게 전달
    if visual_level == 2:
        args.guide_text = "White: AI's plan / Blue: AI's prediction of your plan"
    elif visual_level == 1:
        args.guide_text = "Top: AI's prediction of your plan / Bottom: AI's plan"
    else:
        args.guide_text = ""

    mdp = OvercookedGridworld.from_grid(["XXXXPXXXX", "O       D", "X      2X", "X   1   X", "XXXXSXXXX"])
    env = OvercookedEnv(mdp=mdp, horizon=9999) # horizon은 무한히 둘 수 있도록 9999 유지
    vis = StateVisualizer()
    
    ai = None
    if multi:
        ai = make_agent('EIRAAsync', mdp, "tutorial_grid", K=1)
        ai.set_agent_index(1)
        ai.async_mode = is_async
        ai.teammate_intentions_dict = {}
        ai.current_timestep = 0
        ai.reset()
    
    start_t = time.time()
    score, step, col = run_tutorial(env, screen, vis, ai_agent=ai, args=args)
    log_summary_score(pid, 0, ("Phase1", "tutorial_grid"), -100 - step_id, log_name, score, step, col, time.time() - start_t)

def run_main_step(pid, layout, is_async, visual_level, cond_name, title, description, screen):
    # 1. 시작 대기 화면 (U 키 대기)
    desc_text = f"{description} | " if description else ""
    vu.draw_centered_text(screen, f"[{title}]", f"{desc_text}Press 'U' to start", color=(0, 255, 100))
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_u: waiting = False
        # 무한 루프 과부하 방지를 위한 10ms 대기
        pygame.time.delay(10)
            
    # 2. 로딩 화면 표시
    vu.draw_centered_text(screen, "Please wait...", "Initializing main experiment environment and AI...", color=(255, 200, 0))
    pygame.display.flip()

    # 게임 중에 띄울 설명 텍스트 설정
    guide_text = ""
    if visual_level == 2:
        guide_text = "White: AI's plan / Blue: AI's prediction of your plan"
    elif visual_level == 1:
        guide_text = "Top: AI's prediction of your plan / Bottom: AI's plan"

    start_t = time.time()
    
    score, step, col = run_overcooked_game({
        'layout': layout, 'async': is_async, 'visual_level': visual_level, 'name': cond_name, 
        'log_dir': f"experiments/PID_{pid}/logs", 'episode': 1, 'horizon': 400, 
        'p0': 'Human', 'p1': 'EIRAAsync', 'show_intention': True, 'render': True,
        'guide_text': guide_text
    }, surface=screen)
    log_summary_score(pid, 0, ("Phase1", layout), -1 if "Baseline" in cond_name else -2, cond_name, score, step, col, time.time() - start_t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    # 💡 [핵심 추가] --type 인자 추가 (single, sync, async 중 택 1)
    parser.add_argument('--type', type=str, required=True, choices=['single', 'sync', 'async'], help="Experiment type: single, sync, or async")
    
    args = parser.parse_args()
    pid = args.pid
    exp_type = args.type
    
    pygame.init()
    screen = pygame.display.set_mode((900, 750))
    pygame.display.set_caption(f"Phase 1 - PID: {pid} ({exp_type})")
    
    # 전체 튜토리얼 스텝 정의 (주석 해제)
    tut_steps = [
        (False, False, 0, "Non-Real-time", "", "Tut_S_NRT"), 
        (False, True, 0, "Real-time", "", "Tut_S_RT"), 
        (True, False, 2, "Multi Non-Real-time (H)", "", "Tut_M_NRT_H"), 
        (True, False, 1, "Multi Non-Real-time (N)", "", "Tut_M_NRT_N"), 
        (True, True, 2, "Multi Real-time (H)", "Real-time AI Highlight", "Tut_M_RT_H"), 
        (True, True, 1, "Multi Real-time (N)", "Real-time AI Bubble", "Tut_M_RT_N")
    ]
    
    # 💡 [핵심 추가] 인자에 따라 실행할 튜토리얼 스텝 슬라이싱
    if exp_type == 'single':
        active_steps = tut_steps[0:2]
    elif exp_type == 'sync':
        active_steps = tut_steps[2:4]
    elif exp_type == 'async':
        active_steps = tut_steps[4:6]
    
    # 선택된 튜토리얼 스텝만 실행
    for i, (m, a, v, title, desc, ln) in enumerate(active_steps):
        run_tutorial_step(pid, m, a, v, "tutorial_grid", title, desc, ln, i+1, screen)
    
    # 💡 [핵심 추가] 본 게임(run_main_step)은 sync나 async일 때만 실행
    if exp_type in ['sync', 'async']:
        #run_main_step(pid, "simple", True, 2, "Phase1_Practice", "Practice Session", "Practice with AI highlights on this map", screen)
        run_main_step(pid, "cramped_room", False, 0, "Phase1_Baseline", "Baseline Measurement", "Collaborate without any AI info", screen)
    
    pygame.quit()