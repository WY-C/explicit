import pygame
import time
import sys
import os
import csv
from argparse import ArgumentParser
from main import main as run_overcooked_game, get_parser, TARGET_SCORES

# 💡 [핵심 수정] 3개 맵의 6가지 순열을 정의하되, 
# 인덱스 i와 i+3이 서로 '완전한 역순(Reverse)'이 되도록 배치합니다.
# 이렇게 하면 동일 참가자가 async와 sync에서 겪는 맵 전환(A->B 등)이 절대 겹치지 않습니다.
BALANCED_MAP_ORDERS_6 = [
    [0, 1, 2], # 0: A -> B -> C
    [1, 2, 0], # 1: B -> C -> A
    [2, 0, 1], # 2: C -> A -> B
    [2, 1, 0], # 3: C -> B -> A (0번의 역순)
    [0, 2, 1], # 4: A -> C -> B (1번의 역순)
    [1, 0, 2]  # 5: B -> A -> C (2번의 역순)
]

MAP_POOL = {'A': 'asymmetric_advantages', 'B': 'coordination_ring', 'C': 'counter_circuit'}

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

# 5개 조건을 위한 5x5 라틴 방진
LATIN_SQUARE_5 = [
    [0, 1, 4, 2, 3],
    [1, 2, 0, 3, 4],
    [2, 3, 1, 4, 0],
    [3, 4, 2, 0, 1],
    [4, 0, 3, 1, 2]
]

def get_experimental_plan(pid, env_type):
    map_keys = ['A', 'B', 'C']
    
    # 💡 맵 순서 오프셋 (sync일 경우 완전히 역순인 인덱스(+3)를 타게 됨)
    map_offset = 0 if env_type == 'async' else 3
    map_index = (pid - 1 + map_offset) % 6
    
    selected_map_keys = [map_keys[i] for i in BALANCED_MAP_ORDERS_6[map_index]]
    
    plan = []
    
    # 표현 방식(Condition) 순서 자체를 비틀어주기 위한 오프셋 (sync는 3칸 밀어버림)
    cond_order_offset = 0 if env_type == 'async' else 3
    base_start_cond = (pid - 1 + cond_order_offset) % 5 
    
    # env_type에 따라 사용할 조건 ID 딕셔너리 그룹(0~4 vs 5~9) 결정
    cond_id_group_offset = 0 if env_type == 'async' else 5
    
    for block_idx, m_key in enumerate(selected_map_keys):
        for c_id in LATIN_SQUARE_5[(base_start_cond + block_idx) % 5]:
            actual_cond_id = c_id + cond_id_group_offset
            plan.append({"block": block_idx + 1, "layout_key": m_key, "layout_name": MAP_POOL[m_key], "cond_id": actual_cond_id, "map": m_key})
    return plan

# CSV 헤더에 Collisions 기록
def log_summary_score(pid, block, layout, cond_id, cond_name, score, timestep, col, duration):
    log_dir = f"experiments/PID_{pid}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    file_path = os.path.join(log_dir, f"summary_PID_{pid}.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["PID", "Block", "Map_Key", "Map_Name", "Condition_ID", "Condition_Name", "Score", "Timestep", "Collisions", "Duration_Sec", "Timestamp"])
        writer.writerow([pid, block, layout[0], layout[1], cond_id, cond_name, score, timestep, col, f"{duration:.2f}", time.strftime("%Y-%m-%d %H:%M:%S")])

def wait_for_user(surface, title, subtitle, target_score=None, is_async=None):
    pygame.font.init()
    f40 = pygame.font.SysFont("malgungothic", 40)
    f30 = pygame.font.SysFont("malgungothic", 30)
    f24 = pygame.font.SysFont("malgungothic", 24)
    surface.fill((30, 30, 30))
    surface.blit(f40.render(title, True, (255, 255, 255)), (450 - f40.size(title)[0]//2, 200))
    surface.blit(f30.render(subtitle, True, (0, 255, 255)), (450 - f30.size(subtitle)[0]//2, 280))
    y = 350
    if target_score: 
        surface.blit(f24.render(f"🎯 목표: {target_score}점", True, (255, 200, 0)), (450 - f24.size(f"🎯 목표: {target_score}점")[0]//2, y))
        y += 40
    if is_async is not None:
        txt = "⚡ 실시간 환경" if is_async else "⏳ 비실시간 환경"
        surface.blit(f24.render(txt, True, (100, 255, 100)), (450 - f24.size(txt)[0]//2, y))

    surface.blit(f30.render("'U' 키를 누르세요.", True, (180, 180, 180)), (450 - f30.size("'U' 키를 누르세요.")[0]//2, 500))
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_u: 
                waiting = False
                
                # U키를 누른 직후: 화면을 비우고 '준비 중' 텍스트 표시
                surface.fill((30, 30, 30))
                loading_text = "준비 중입니다... 잠시만 기다려주세요."
                surface.blit(f40.render(loading_text, True, (255, 255, 255)), (450 - f40.size(loading_text)[0]//2, 300))
                pygame.display.flip()
                
                # OS에 이벤트 처리 상태를 업데이트하여 '응답 없음' 방지
                pygame.event.pump()

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--pid', type=int, required=True)
    # 실험 환경 타입 인자 추가 (필수)
    parser.add_argument('--study_type', type=str, choices=['async', 'sync'], required=True, help="실험 환경 선택: 'async'(0~4번 조건) 또는 'sync'(5~9번 조건)")
    # 이어서 시작할 수 있도록 trial 인자 추가 (기본값 1)
    parser.add_argument('--trial', type=int, default=1, help="특정 세션(1부터 시작)부터 실험을 시작합니다.")
    
    args = parser.parse_args()
    pid = args.pid
    env_type = args.study_type
    start_trial = args.trial
    
    study_plan = get_experimental_plan(pid, env_type)
    total_sessions = len(study_plan) # 블록당 5개 조건 * 3블록 = 총 15세션
    
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    
    # 지정된 trial부터 루프 시작 (i는 원래 세션의 인덱스를 유지)
    for i in range(start_trial-1, total_sessions):
        session = study_plan[i]
        cond = CONDITION_SETTINGS[session['cond_id']]
        target = TARGET_SCORES.get(session['layout_name'], 60)
        
        # 화면에 표시되는 세션 번호를 {i+1}로 변경하여 사용자가 입력한 trial과 시각적으로 일치시킴
        wait_for_user(screen, f"PID: {pid} ({env_type.upper()}) | 세션 {i+1}/{total_sessions}", f"조건: {session['map']} - {session['cond_id']}", target, cond['async'])

        # main.py가 반환하는 실제 플레이 시간(duration)을 언패킹
        score, step, col, duration = run_overcooked_game({
            'layout': session['layout_name'], 'visual_level': cond['vis'], 'show_intention': cond['show_intention'],
            'async': cond['async'], 'log_dir': f"experiments/PID_{pid}/logs", 'name': cond['name'], 'episode': 1, 'horizon': 400,
            'p0': 'Human', 'p1': 'EIRAAsync', 'cook_time': 20, 'timestep': 400, 'gpt_model': 'Qwen/Qwen3-VL-8B-Instruct', 'prompt_level': 'l3-aip'
        }, surface=screen)
        
        # 받아온 duration을 CSV에 기록
        log_summary_score(pid, session['block'], (session['layout_key'], session['layout_name']), session['cond_id'], cond['name'], score, step, col, duration)

        # 블록 하나당 조건이 5개이므로, 5의 배수마다 휴식 (마지막 세션 제외)
        if (i + 1) % 5 == 0 and (i + 1) < total_sessions: 
            wait_for_user(screen, "블록 종료", "")
        elif (i+1) == total_sessions:
            wait_for_user(screen, "실험 종료", "모든 세션이 완료되었습니다. 참여해주셔서 감사합니다!")
        
            
    pygame.quit()