import sys
import os
import time
import csv
import argparse
import pygame
from main import get_parser, main as run_overcooked_game
from tutorial import run_tutorial

# 💡 CSV 헤더에 Inter_Collisions 추가
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

def run_tutorial_step(pid, multi, is_async, visual_level, layout, title, description, log_name, screen):
    print(f"\n🎯 [튜토리얼] {title}\n💡 {description}"); input("👉 [Enter] 시작...")
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
    from utils import make_agent

    class Args: pass
    args = Args()
    args.is_async, args.visual_level, args.delay, args.log_dir, args.name = is_async, visual_level, 400, f"experiments/PID_{pid}/logs", log_name

    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv(mdp=mdp, horizon=9999); vis = StateVisualizer()
    ai = make_agent('EIRAAsync', mdp, layout, K=1) if multi else None
    if ai: ai.set_agent_index(1)

    start_t = time.time()
    score, step, inter_col = run_tutorial(env, screen, vis, ai_agent=ai, args=args)
    log_summary_score(pid, 0, ("Phase1", layout), -100, log_name, score, step, inter_col, time.time() - start_t)

def run_main_step(pid, layout, is_async, visual_level, cond_name, title, description, screen):
    print(f"\n🎯 [적응] {title}\n💡 {description}"); input("👉 [Enter] 시작...")
    start_t = time.time()
    # 💡 main.py에서 score, step, inter_col을 모두 받음
    score, step, inter_col = run_overcooked_game({
        'layout': layout, 'async': is_async, 'visual_level': visual_level, 'name': cond_name, 
        'log_dir': f"experiments/PID_{pid}/logs", 'episode': 1, 'horizon': 400, 'p0': 'Human', 
        'p1': 'EIRAAsync', 'cook_time': 20, 'timestep': 400, 'render': True,
        'gpt_model': 'Qwen/Qwen3-VL-8B-Instruct', 'prompt_level': 'l3-aip', 'show_intention': True
    }, surface=screen)
    log_summary_score(pid, 0, ("Phase1", layout), -1, cond_name, score, step, inter_col, time.time() - start_t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args(); pid = args.pid
    pygame.init(); screen = pygame.display.set_mode((900, 750))
        # 💡 튜토리얼 스텝들도 CSV에 기록됩니다.
    run_tutorial_step(pid, False, False, 0, "tutorial_grid", "싱글 비실시간", "조작법 연습", "Tut_Single_NRT", screen)
    run_tutorial_step(pid, False, True, 0, "tutorial_grid", "싱글 실시간", "실시간 감각 연습", "Tut_Single_RT", screen)
    run_tutorial_step(pid, True, False, 2, "tutorial_grid", "멀티 비실시간(H)", "하이라이트 확인", "Tut_Multi_NRT_High", screen)
    run_tutorial_step(pid, True, False, 1, "tutorial_grid", "멀티 비실시간(N)", "자연어 확인", "Tut_Multi_NRT_NL", screen)
    run_tutorial_step(pid, True, True, 2, "tutorial_grid", "멀티 실시간(H)", "실시간 하이라이트", "Tut_Multi_RT_High", screen)
    run_tutorial_step(pid, True, True, 1, "tutorial_grid", "멀티 실시간(N)", "실시간 자연어", "Tut_Multi_RT_NL", screen)
    # run_tutorial_step(...) 및 run_main_step(...) 호출 부분
    run_main_step(pid, "cramped_room", True, 2, "Phase1_Practice", "연습", "cramped_room 적응", screen)
    run_main_step(pid, "cramped_room", True, 0, "Phase1_Baseline", "베이스라인", "정보 없음", screen)
    
    pygame.quit()
    print(f"\n🎉 PID {pid} 완료! 모든 충돌 데이터가 CSV에 저장되었습니다.")