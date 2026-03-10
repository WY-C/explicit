import pygame
import time
import sys
import os
import csv
from argparse import ArgumentParser
from main import main as run_overcooked_game, get_parser, TARGET_SCORES

LATIN_SQUARE_MAPS = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
MAP_POOL = {'A': 'cramped_room', 'B': 'asymmetric_advantages', 'C': 'coordination_ring', 'D': 'counter_circuit'}

CONDITION_SETTINGS = {
    0: {"async": True,  "vis": 0, "show_intention": False, "name": "RT_None"},
    1: {"async": True,  "vis": 2, "show_intention": False, "name": "RT_High_Plan"},
    2: {"async": True,  "vis": 2, "show_intention": True,  "name": "RT_High_Plan_Infer"},
    3: {"async": True,  "vis": 1, "show_intention": False, "name": "RT_NL_Plan"},
    4: {"async": True,  "vis": 1, "show_intention": True,  "name": "RT_NL_Plan_Infer"},
    5: {"async": False, "vis": 0, "show_intention": False, "name": "NRT_None"},
    6: {"async": False, "vis": 2, "show_intention": False, "name": "NRT_High_Plan"},
    7: {"async": False, "vis": 2, "show_intention": True,  "name": "NRT_High_Plan_Infer"},
    8: {"async": False, "vis": 1, "show_intention": False, "name": "NRT_NL_Plan"},
    9: {"async": False, "vis": 1, "show_intention": True,  "name": "NRT_NL_Plan_Infer"},
}

LATIN_SQUARE_10 = [
    [0, 1, 9, 2, 8, 3, 7, 4, 6, 5], [1, 2, 0, 3, 9, 4, 8, 5, 7, 6],
    [2, 3, 1, 4, 0, 5, 9, 6, 8, 7], [3, 4, 2, 5, 1, 6, 0, 7, 9, 8],
    [4, 5, 3, 6, 2, 7, 1, 8, 0, 9], [5, 6, 4, 7, 3, 8, 2, 9, 1, 0],
    [6, 7, 5, 8, 4, 9, 3, 0, 2, 1], [7, 8, 6, 9, 5, 0, 4, 1, 3, 2],
    [8, 9, 7, 0, 6, 1, 5, 2, 4, 3], [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]
]

def get_experimental_plan(pid):
    map_keys = ['A', 'B', 'C', 'D']
    selected_map_keys = [map_keys[i] for i in LATIN_SQUARE_MAPS[(pid - 1) % 4][:3]]
    plan = []
    base_start_cond = (pid - 1) % 10 
    for block_idx, m_key in enumerate(selected_map_keys):
        for c_id in LATIN_SQUARE_10[(base_start_cond + block_idx) % 10]:
            plan.append({"block": block_idx + 1, "layout_key": m_key, "layout_name": MAP_POOL[m_key], "cond_id": c_id})
    return plan

# 💡 CSV 헤더에 P0_Collisions, P1_Collisions 추가
def log_summary_score(pid, block, layout, cond_id, cond_name, score, timestep, p0_col, p1_col, duration):
    log_dir = f"experiments/PID_{pid}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    file_path = os.path.join(log_dir, f"summary_PID_{pid}.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["PID", "Block", "Map_Key", "Map_Name", "Condition_ID", "Condition_Name", "Score", "Timestep", "P0_Collisions", "P1_Collisions", "Duration_Sec", "Timestamp"])
        writer.writerow([pid, block, layout[0], layout[1], cond_id, cond_name, score, timestep, p0_col, p1_col, f"{duration:.2f}", time.strftime("%Y-%m-%d %H:%M:%S")])

def wait_for_user(surface, title, subtitle, target_score=None, is_async=None):
    pygame.font.init()
    f40 = pygame.font.SysFont("malgungothic", 40); f30 = pygame.font.SysFont("malgungothic", 30); f24 = pygame.font.SysFont("malgungothic", 24)
    surface.fill((30, 30, 30))
    surface.blit(f40.render(title, True, (255, 255, 255)), (450 - f40.size(title)[0]//2, 200))
    surface.blit(f30.render(subtitle, True, (0, 255, 255)), (450 - f30.size(subtitle)[0]//2, 280))
    y = 350
    if target_score: surface.blit(f24.render(f"🎯 목표: {target_score}점", True, (255, 200, 0)), (450 - f24.size(f"🎯 목표: {target_score}점")[0]//2, y)); y+=40
    if is_async is not None:
        txt = "⚡ 실시간 환경" if is_async else "⏳ 비실시간 환경"
        surface.blit(f24.render(txt, True, (100, 255, 100)), (450 - f24.size(txt)[0]//2, y))
    surface.blit(f30.render("'U' 키를 누르면 시작합니다.", True, (180, 180, 180)), (450 - f30.size("'U' 키를 누르면 시작합니다.")[0]//2, 500))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_u: waiting = False

if __name__ == '__main__':
    parser = get_parser(); parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args(); pid = args.pid
    study_plan = get_experimental_plan(pid); pygame.init(); screen = pygame.display.set_mode((900, 600))
    
    for i, session in enumerate(study_plan):
        cond = CONDITION_SETTINGS[session['cond_id']]
        target = TARGET_SCORES.get(session['layout_name'], 60)
        wait_for_user(screen, f"PID: {pid} | 세션 {i+1}/30", f"맵: {session['layout_name']}", target, cond['async'])
        
        start_time = time.time()
        score, step, p0_col, p1_col = run_overcooked_game({
            'layout': session['layout_name'], 'visual_level': cond['vis'], 'show_intention': cond['show_intention'],
            'async': cond['async'], 'log_dir': f"experiments/PID_{pid}/logs", 'name': cond['name'], 'episode': 1, 'horizon': 400,
            'p0': 'Human', 'p1': 'EIRAAsync', 'cook_time': 20, 'timestep': 400, 'gpt_model': 'Qwen/Qwen3-VL-8B-Instruct', 'prompt_level': 'l3-aip'
        }, surface=screen)
        
        log_summary_score(pid, session['block'], (session['layout_key'], session['layout_name']), session['cond_id'], cond['name'], score, step, p0_col, p1_col, time.time() - start_time)

        if (i + 1) % 10 == 0 and (i + 1) < 30: wait_for_user(screen, "블록 종료", "잠시 휴식 후 진행하세요.")
    pygame.quit()