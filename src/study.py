#todo 맵에도 LATIN SQUARE 적용
import pygame
import time
import sys
import os
import csv
from argparse import ArgumentParser
from main import main as run_overcooked_game, get_parser
# 0: 'A', 1: 'B', 2: 'C', 3: 'D'
LATIN_SQUARE_MAPS = [
    [0, 1, 3, 2],
    [1, 2, 0, 3],
    [2, 3, 1, 0],
    [3, 0, 2, 1]
]

MAP_POOL = {'A': 'cramped_room', 'B': 'asymmetric_advantages', 'C': 'coordination_ring', 'D': 'counter_circuit'}
CONDITION_SETTINGS = {
    0: {"async": True,  "vis": 0, "name": "RT_None"},
    1: {"async": True,  "vis": 3, "name": "RT_High_Plan"},
    2: {"async": True,  "vis": 3, "name": "RT_High_Plan_Infer"},
    3: {"async": True,  "vis": 2, "name": "RT_NL_Plan"},
    4: {"async": True,  "vis": 2, "name": "RT_NL_Plan_Infer"},
    5: {"async": False, "vis": 0, "name": "NRT_None"},
    6: {"async": False, "vis": 3, "name": "NRT_High_Plan"},
    7: {"async": False, "vis": 3, "name": "NRT_High_Plan_Infer"},
    8: {"async": False, "vis": 2, "name": "NRT_NL_Plan"},
    9: {"async": False, "vis": 2, "name": "NRT_NL_Plan_Infer"},
}

LATIN_SQUARE_10 = [
    [0, 1, 9, 2, 8, 3, 7, 4, 6, 5],
    [1, 2, 0, 3, 9, 4, 8, 5, 7, 6],
    [2, 3, 1, 4, 0, 5, 9, 6, 8, 7],
    [3, 4, 2, 5, 1, 6, 0, 7, 9, 8],
    [4, 5, 3, 6, 2, 7, 1, 8, 0, 9],
    [5, 6, 4, 7, 3, 8, 2, 9, 1, 0],
    [6, 7, 5, 8, 4, 9, 3, 0, 2, 1],
    [7, 8, 6, 9, 5, 0, 4, 1, 3, 2],
    [8, 9, 7, 0, 6, 1, 5, 2, 4, 3],
    [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]
]

def get_experimental_plan(pid):
    map_keys = ['A', 'B', 'C', 'D']
    
    # PID에 따라 사용할 맵의 라틴 방진 행(Row)을 결정합니다.
    map_row_idx = (pid - 1) % len(LATIN_SQUARE_MAPS)
    
    # 해당 행의 순서대로 맵 인덱스를 가져오고, 앞서 말씀하신 대로 총 3개의 블록만 진행하므로 [:3]으로 3개만 자릅니다.
    selected_map_indices = LATIN_SQUARE_MAPS[map_row_idx][:3]
    selected_map_keys = [map_keys[i] for i in selected_map_indices]
    
    plan = []
    
    # PID에 따라 첫 번째 블록에서 사용할 에이전트 조건 라틴 방진의 행 번호 결정
    base_start_cond = (pid - 1) % 10 
    
    for block_idx, m_key in enumerate(selected_map_keys):
        block_layout = MAP_POOL[m_key]
        
        # 블록이 넘어갈 때마다 에이전트 조건 라틴 방진의 다음 행(Row)을 사용
        row_idx = (base_start_cond + block_idx) % 10
        block_conditions = LATIN_SQUARE_10[row_idx]
        
        for c_id in block_conditions:
            plan.append({
                "block": block_idx + 1, 
                "layout_key": m_key, 
                "layout_name": block_layout, 
                "cond_id": c_id
            })
            
    return plan

def log_summary_score(pid, block, layout, cond_id, cond_name, score):
    log_dir = f"experiments/PID_{pid}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    file_path = os.path.join(log_dir, f"summary_PID_{pid}.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["PID", "Block", "Map_Key", "Map_Name", "Condition_ID", "Condition_Name", "Score", "Timestamp"])
        writer.writerow([pid, block, layout[0], layout[1], cond_id, cond_name, score, time.strftime("%Y-%m-%d %H:%M:%S")])

def wait_for_user(surface, title, subtitle):
    if not pygame.font.get_init(): pygame.font.init()
    try: font = pygame.font.SysFont("malgungothic", 40); small_font = pygame.font.SysFont("malgungothic", 30)
    except: font = pygame.font.SysFont("arial", 40); small_font = pygame.font.SysFont("arial", 30)
    
    pygame.event.clear()
    waiting = True
    while waiting:
        surface.fill((30, 30, 30))
        t_surf = font.render(title, True, (255, 255, 255))
        s_surf = small_font.render(subtitle, True, (0, 255, 255))
        
        # 화면에 표시되는 안내 문구를 변경했습니다.
        p_surf = small_font.render("'U' 키를 누르면 시작합니다.", True, (180, 180, 180)) 
        
        surface.blit(t_surf, t_surf.get_rect(center=(450, 250)))
        surface.blit(s_surf, s_surf.get_rect(center=(450, 350)))
        surface.blit(p_surf, p_surf.get_rect(center=(450, 500)))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                # 아무 키가 아닌, 'U' 키(pygame.K_u)를 눌렀을 때만 waiting을 False로 바꿉니다.
                if event.key == pygame.K_u: 
                    waiting = False
                    
        pygame.time.delay(10)
if __name__ == '__main__':
    # 1. 먼저 사용자가 입력한 인자(PID)를 받아옵니다.
    parser = get_parser()
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    base_variant = vars(args)
    pid = base_variant['pid']
    
    # 2. 받아온 PID를 이용해 미리 폴더를 안전하게 만들어 둡니다.
    log_dir = f"experiments/PID_{pid}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    
    # 3. 실험 계획(라틴 방진)을 짜고 파이게임을 시작합니다.
    study_plan = get_experimental_plan(pid)
    
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    
    for i, session in enumerate(study_plan):
        cond_info = CONDITION_SETTINGS[session['cond_id']]
        
        # 1. 파이게임 윈도우 창의 상단 이름(캡션)을 동적으로 변경합니다.
        window_title = f"PID: {pid} | {session['layout_key']}-{session['cond_id']} (세션 {i+1}/30)"
        pygame.display.set_caption(window_title)
        
        # 2. 대기 화면 중앙에도 PID와 세션 정보, 현재 조건을 깔끔하게 띄워줍니다.
        wait_for_user(screen, f"PID: {pid} | 세션 {i+1} / 30", f"현재 조건: {session['layout_key']}-{session['cond_id']}")
        
        current_variant = base_variant.copy()
        current_variant.update({
            'layout': session['layout_name'], 'visual_level': cond_info['vis'],
            'async': cond_info['async'], 'log_dir': f"experiments/PID_{pid}/logs", 'episode': 1
        })
        
        try:
            score = run_overcooked_game(current_variant, surface=screen)
            log_summary_score(pid, session['block'], (session['layout_key'], session['layout_name']), session['cond_id'], cond_info['name'], score)
        except Exception as e: print(f"Error: {e}")

        if (i + 1) % 10 == 0 and (i + 1) < 30:
            # 블록(10판)이 끝날 때 창 이름도 휴식 상태로 바꿔줍니다.
            pygame.display.set_caption(f"PID: {pid} | 휴식 시간")
            wait_for_user(screen, "블록 종료", "잠시 휴식 후 진행하세요.")

    pygame.display.set_caption("실험 종료")
    wait_for_user(screen, "실험 종료", "수고하셨습니다.")
    pygame.quit()
