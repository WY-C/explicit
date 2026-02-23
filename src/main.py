"""
=todo=

본인의 계획만 출력하는 경우 있어야함.
coordination_ring, counter_circuit, cramped_room 프롬프트 제작하기
마지막 명령 후 3초뒤에 호출이 없으면, 새로하기
문제: 어떤게 가까운 것인지 알 지 못하는 것 같음. : 프롬프트에서 각 object와 얼마나 거리가 있는지에 대한 정보 주기.
-> 각 object의 위치를 주지 말고, 현재 위치에서 각 object가 얼마나 떨어져 있는지를 주기.
벤치마크 진행하기
=연구=
    불확실성?
=문제점=
    LLM이 반대쪽에 ex, 초록색 구역이 아닌곳에 초록색으로 의도를 파악하는 경우가 있음.
=궁금증=
    proagent는 counter_circuit에서 협력하는가?
    - 예시를 구체적으로 주어야 하는가?
=벤치마크=
    내 환경의 ProAgent와 비교
=논문=
    ProAgent가 문단의 첫 문장에 나올 것이 아니라, detail에 조금만 들어가야 한다.
=진행한 것들=
    파랑머리가 이야기하는 것처럼 (✅ 렌더링 함수에 구현됨)
    파랑머리: 의도, 계획
    이동 경로를 모두 저장했다가 보여주기 -> 이전 3개 정도만 보여주기
    프롬프트 예시 추가하기

참고 논문 및 아이디어
    LLM-Powered Hierarchical Language Agent for Real-time Human-AI Coordination
        2.5Hz 기본, 3.5Hz 게임플레이 박진감. -> 유저의 평균속도에 agent가 맞춰서 행동하는 거였음.
        100초 진행. -> 250step -> 근데 이거는 제한시간에 task를 완료하는거라서 좀 애매함.
    가장 가까운 곳이 아니라, LLM이 어떤 pot에 넣을지까지 정하기 (✅ 방금 action(index) 파싱 및 하이라이트로 구현 완료)
참고 논문 2
    Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration
        In the real-time settings, each timestep corresponds to 0.25 seconds in the real world.
        500timestep / 250ms -> 125초

수정사항들
    cook time argument -> 요리가 완료됨을 추가하는 argument + layout가서 따로 수정해주기
    IntentionResponsiveAgent
"""

import time
import datetime
import os
import json
from argparse import ArgumentParser
import numpy as np
import pygame

# Rich 라이브러리 임포트
from rich import print as rprint
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn
)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*") 

from distutils.util import strtobool
def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

import importlib_metadata
try:
    VERSION = importlib_metadata.version("overcooked_ai")
except:
    VERSION = "0.0.1"

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

print(f'\n----This overcook version is {VERSION}----\n')

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action

from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent
from overcooked_ai_py.agents.agent import Agent
import re

# [HumanAgent Class]
class HumanAgent(Agent):
    def __init__(self):
        super().__init__()
        self.next_action = None 

    def set_next_action(self, action):
        """키 입력 이벤트에서 호출되어 행동을 저장함"""
        self.next_action = action

    def action(self, state):
        """환경이 step을 진행할 때 호출됨. 저장된 행동을 반환하고 초기화."""
        if self.next_action is not None:
            a = self.next_action
            self.next_action = None
            return a
        return (0,0)

# ======================================================================================
# [Helper] 화면 중앙 텍스트 출력 (카운트다운용)
# ======================================================================================
def parse_separate_highlights(thought_text, layout_dict, num_AI=None):
    """
    my_player_id (int, optional): 0 또는 1. 지정할 경우, Plan이 해당 플레이어 ID인지 검증합니다.
    """
    highlight_for_inference_coords = []
    highlight_for_plan_coords = []
    
    if not thought_text or not layout_dict:
        return [], []

    def _action_to_coords(act_str):
        # (기존과 동일하여 생략)
        if not act_str: return []
        key_map = {
            "pickup_onion": "onion_dispenser",
            "pickup_dish": "dish_dispenser",
            "put_onion_in_pot": "pot",
            "fill_dish_with_soup": "pot",
            "deliver_soup": "serving",
            "place_obj_on_counter": "counter" # counter 추가 필요시
        }
        match = re.search(r'(\w+)\(?(\d+)?\)?', act_str) # 인자 없는 경우 대비 수정
        if match:
            act_name = match.group(1)
            idx_str = match.group(2)
            
            # 인자가 없는 스킬(wait, place_obj 등) 처리
            if not idx_str: 
                return [] 
                
            idx = int(idx_str)
            target_key = key_map.get(act_name)
            
            if target_key and target_key in layout_dict:
                items = layout_dict[target_key]
                if 0 <= idx < len(items):
                    target = items[idx]
                    if isinstance(target, dict) and 'position' in target:
                        return [target['position']]
                    else:
                        return [target]
        return []

    # --- Intention (상대방) 파싱 ---
    # 예: Intention for Player 0: "..."
    # 숫자(\d+)를 캡처해서 파트너인지 확인 가능
    intention_match = re.search(r'Intention.*?(?:Player (\d+))?.*:\s*"([^"]+)"', thought_text, re.IGNORECASE)
    if intention_match:
        # 그룹 2가 액션 문자열 (그룹 1은 플레이어 번호)
        # 플레이어 번호 검증 로직을 넣을 수도 있음 (생략 가능)
        action_str = intention_match.group(2)
        highlight_for_inference_coords = _action_to_coords(action_str)

    # --- Plan (나) 파싱 ---
    # 예: Plan for Player 1: "..."
    plan_match = re.search(r'Plan.*?(?:Player (\d+))?.*:\s*"([^"]+)"', thought_text, re.IGNORECASE)
    if plan_match:
        parsed_id = plan_match.group(1) # '0' 또는 '1'
        action_str = plan_match.group(2)
        
        # 검증 로직: 내 ID가 주어졌는데, 텍스트의 ID와 다르면 무시하거나 경고
        if num_AI is not None and parsed_id is not None:
            if int(parsed_id) != num_AI:
                print(f"Warning: Parsed plan for Player {parsed_id}, but I am Player {num_AI}.")
                # 필요 시 return [], [] 혹은 에러 처리
        
        highlight_for_plan_coords = _action_to_coords(action_str)

    return highlight_for_inference_coords, highlight_for_plan_coords


#처음 시작
def draw_centered_text(window, text, sub_text=None, color=(0, 0, 0), bg_color=(255, 255, 255)):
    """화면 중앙에 텍스트를 그립니다."""
    if window is None: return
    
    window.fill(bg_color)
    screen_width, screen_height = window.get_size()
    
    # 폰트 설정 (한글 폰트 우선, 없으면 기본)
    font_name = "malgungothic" if "malgungothic" in pygame.font.get_fonts() else None
    main_font = pygame.font.SysFont(font_name, 50, bold=True)
    sub_font = pygame.font.SysFont(font_name, 30)

    # 메인 텍스트
    text_surf = main_font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(screen_width // 2, screen_height // 2 - 20))
    window.blit(text_surf, text_rect)
    
    # 서브 텍스트 (옵션)
    if sub_text:
        sub_surf = sub_font.render(sub_text, True, (100, 100, 100))
        sub_rect = sub_surf.get_rect(center=(screen_width // 2, screen_height // 2 + 40))
        window.blit(sub_surf, sub_rect)
        
    pygame.display.flip()

# ======================================================================================
# [Unified Render Function] 통합 렌더링 함수
# visual_level 1: Emoji Mode
# visual_level 2: Natural Language Mode
# visual_level 3: Highlight Mode
# ======================================================================================
def render_game(window, visualizer, env, step, horizon, reward, num_AI, visual_level, layout_dict,
                thought_msg=None,):

    highlight_color_green = (80, 220, 150)
    highlight_color_blue = (50, 120, 255)
    highlight_color_purple = (32, 170, 210)
    
    if not window or not visualizer:
        return

    # 1. 배경 및 공통 설정
    window.fill((255, 255, 255)) 
    screen_width, screen_height = window.get_size()
    
    font_name = "malgungothic" if "malgungothic" in pygame.font.get_fonts() else None
    font_header = pygame.font.SysFont(font_name, 30)
    
    # 상단 헤더 (Step, Reward)
    text = font_header.render(f"Step: {step}/{horizon} | Reward: {reward}", True, (0, 0, 0))
    window.blit(text, (10, 10))

    # -----------------------------------------------------------
    # 2. 생각(Thought) 시각화 분기
    # -----------------------------------------------------------
    current_y = 60  # 텍스트/아이콘 시작 Y 위치
    left_margin = 10
    map_start_y = max(current_y + 20, 170)
    
    state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
    surf_width, surf_height = state_surface.get_size()
    
    start_x = (screen_width - surf_width) // 2
    window.blit(state_surface, (start_x, map_start_y))
    if thought_msg:
        # 공통 헬퍼 함수
        def chef_frame_name(d, obj): return f"{d}-{obj}" if obj else d
        def hat_frame_name(d, color): return f"{d}-{color}hat"

        # --- [Visual Level 1: Emoji Mode] ---
        if visual_level == 1:
            bubble_font = pygame.font.SysFont(font_name, 28) 
            text_color = (0, 0, 0)
            icon_size = 45    
            line_height = 45

            icon_map = {
                "fill_dish_with_soup": "soup-onion-cooked",
                "wait": "stay",
                "deliver_soup": "EAST-soup-onion",
                "onion": "onions", "양파": "onions",
                "dish": "dishes", "접시": "dishes",
                "pot": "pot", "냄비": "pot",
                "soup": "soup", "수프": "soup"
            }

            # (1) 셰프 아이콘 그리기
            chef_surf = pygame.Surface((visualizer.UNSCALED_TILE_SIZE, visualizer.UNSCALED_TILE_SIZE), pygame.SRCALPHA)
            visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), chef_frame_name("SOUTH", None))
            hat_color = "blue" if num_AI == 0 else "green"
            visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), hat_frame_name("SOUTH", hat_color))
            
            scaled_chef = pygame.transform.scale(chef_surf, (icon_size, icon_size))
            window.blit(scaled_chef, (left_margin, current_y)) 
            
            # (2) 콜론 그리기
            colon_surf = bubble_font.render(" : ", True, text_color)
            colon_x = left_margin + icon_size 
            colon_y_colon = current_y + (line_height - colon_surf.get_height()) // 2
            window.blit(colon_surf, (colon_x, colon_y_colon))
            text_start_x = colon_x + colon_surf.get_width()

            # (3) 텍스트 파싱 및 아이콘 매핑
            raw_lines = thought_msg.split('\n')
            for line in raw_lines:
                line = line.strip()
                if not line: continue 

                display_text = ""
                if "Intention" in line: display_text = "You want"
                elif "Plan" in line: display_text = "I will do"
                else: continue 

                # 키워드 매칭
                line_lower = line.lower()
                sprite_name = None
                for key, s_name in icon_map.items():
                    if key in line_lower:
                        sprite_name = s_name
                        break 

                # 텍스트 출력
                current_x = text_start_x
                text_surf = bubble_font.render(display_text, True, text_color)
                text_y_pos = current_y + (line_height - text_surf.get_height()) // 2
                window.blit(text_surf, (current_x, text_y_pos))
                current_x += text_surf.get_width() + 10 

                # 아이콘 출력
                if sprite_name:
                    obj_surf = pygame.Surface((visualizer.UNSCALED_TILE_SIZE, visualizer.UNSCALED_TILE_SIZE), pygame.SRCALPHA)
                    
                    if sprite_name == 'stay':
                        try:
                            raw_img = pygame.image.load('stay.png').convert_alpha()
                            resized_img = pygame.transform.scale(raw_img, (visualizer.UNSCALED_TILE_SIZE, visualizer.UNSCALED_TILE_SIZE))
                            obj_surf.blit(resized_img, (0, 0))
                        except FileNotFoundError: pass
                    else:
                        try: visualizer.TERRAINS_IMG.blit_on_surface(obj_surf, (0, 0), sprite_name)
                        except KeyError:
                            try: visualizer.OBJECTS_IMG.blit_on_surface(obj_surf, (0, 0), sprite_name)
                            except KeyError:
                                try: visualizer.CHEFS_IMG.blit_on_surface(obj_surf, (0, 0), sprite_name)
                                except: pass

                    scaled_obj = pygame.transform.scale(obj_surf, (icon_size, icon_size))
                    window.blit(scaled_obj, (current_x, current_y + (line_height - icon_size)//2))

                current_y += line_height

        # --- [Visual Level 2: Natural Language Mode] ---
        elif visual_level == 2:
            font_text = pygame.font.SysFont(font_name, 24)
            icon_size = 45    
            line_height = 40  
            
            # (1) 셰프 아이콘 그리기
            chef_surf = pygame.Surface((visualizer.UNSCALED_TILE_SIZE, visualizer.UNSCALED_TILE_SIZE), pygame.SRCALPHA)
            visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), chef_frame_name("SOUTH", None))
            hat_color = "blue" if num_AI == 0 else "green"
            visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), hat_frame_name("SOUTH", hat_color))
            
            scaled_chef = pygame.transform.scale(chef_surf, (icon_size, icon_size))
            window.blit(scaled_chef, (left_margin, current_y))
            
            # (2) 콜론 그리기
            colon_surf = font_header.render(" : ", True, (0, 0, 0))
            colon_x = left_margin + icon_size
            colon_y_colon = current_y + (icon_size - colon_surf.get_height()) // 2
            window.blit(colon_surf, (colon_x, colon_y_colon))
            text_start_x = colon_x + colon_surf.get_width()

            # (3) 텍스트 정제 및 출력
            if "Intention for Player" in thought_msg:
                thought_msg = thought_msg[thought_msg.find("Intention for Player"):]
                thought_msg = thought_msg.replace("Intention for Player 1", "Intention for Partner")
            if "Plan for Player 0" in thought_msg:
                thought_msg = thought_msg.replace("Plan for Player 0", "\nPlan for Me")

            lines = thought_msg.split('\n')
            text_y_pos = current_y + (icon_size - font_text.get_height()) // 2 # 수직 중앙 정렬
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                
                text_surf = font_text.render(line, True, (0, 0, 0))
                draw_x = text_start_x
                window.blit(text_surf, (draw_x, text_y_pos))
                text_y_pos += line_height
            
            current_y = text_y_pos
        elif visual_level == 3:
            highlight_for_inference_coords, highlight_for_plan_coords = parse_separate_highlights(thought_msg, layout_dict, num_AI=num_AI)

            # 1. 타일 하나의 픽셀 크기 계산
            grid_width = len(env.mdp.terrain_mtx[0])
            grid_height = len(env.mdp.terrain_mtx)

            if grid_width > 0 and grid_height > 0:
                tile_w = surf_width / grid_width
                tile_h = surf_height / grid_height

    
                if num_AI == 0:
                    inf_color = highlight_color_green  
                    plan_color = highlight_color_blue 
                else:
                    inf_color = highlight_color_blue  
                    plan_color = highlight_color_green  

                # 4. 그리기 작업을 리스트로 묶어서 처리
                # (좌표 리스트, 해당 색상) 쌍을 만들어 순회합니다.
                draw_tasks = [
                    (highlight_for_inference_coords, inf_color),
                    (highlight_for_plan_coords, plan_color)
                ]

                for coords_list, color in draw_tasks:
                    # 좌표 리스트가 비어있으면 스킵
                    if not coords_list:
                        continue

                    # 해당 색상의 반투명 서피스 생성
                    s = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                    s.fill((*color, 100)) # Alpha=100 (반투명)

                    for (hx, hy) in coords_list:
                        # 좌표 유효성 검사
                        if 0 <= hx < grid_width and 0 <= hy < grid_height:
                            draw_x = start_x + (hx * tile_w)
                            draw_y = map_start_y + (hy * tile_h)
                            
                            # 1) 반투명 색칠하기
                            window.blit(s, (draw_x, draw_y))
                            
                            # 2) 진한 테두리 그리기
                            rect = pygame.Rect(draw_x, draw_y, tile_w, tile_h)
                            pygame.draw.rect(window, color, rect, 3) # 두께 3
            else:
                if highlight_for_inference_coords:
                    # 1. 타일 하나의 픽셀 크기 계산 (전체 이미지 너비 / 그리드 가로 칸 수)
                    grid_width = len(env.mdp.terrain_mtx[0])
                    grid_height = len(env.mdp.terrain_mtx)
                    
                    if grid_width > 0 and grid_height > 0:
                        tile_w = surf_width / grid_width
                        tile_h = surf_height / grid_height

                        # 2. 하이라이트 전용 서피스 생성 (반투명 지원)
                        # 타일 크기만큼의 투명 종이를 만듭니다.
                        s = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                        
                        # 색상 채우기 (R, G, B, Alpha) -> Alpha=100 (0~255, 숫자가 클수록 불투명)
                        s.fill((*highlight_color_green, 100))

                        for (hx, hy) in highlight_for_inference_coords:
                            # 좌표 유효성 검사 (맵 밖으로 나가는 것 방지)
                            if 0 <= hx < grid_width and 0 <= hy < grid_height:
                                # 그릴 위치 계산
                                draw_x = start_x + (hx * tile_w)
                                draw_y = map_start_y + (hy * tile_h)
                                
                                # 1) 반투명 색칠하기
                                window.blit(s, (draw_x, draw_y))
                                
                                # 2) 진한 테두리 그리기 (선택 사항)
                                rect = pygame.Rect(draw_x, draw_y, tile_w, tile_h)
                                pygame.draw.rect(window, highlight_color_green, rect, 3) # 두께 3

                if highlight_for_plan_coords:
                    # 1. 타일 하나의 픽셀 크기 계산 (전체 이미지 너비 / 그리드 가로 칸 수)
                    grid_width = len(env.mdp.terrain_mtx[0])
                    grid_height = len(env.mdp.terrain_mtx)
                    
                    if grid_width > 0 and grid_height > 0:
                        tile_w = surf_width / grid_width
                        tile_h = surf_height / grid_height

                        # 2. 하이라이트 전용 서피스 생성 (반투명 지원)
                        # 타일 크기만큼의 투명 종이를 만듭니다.
                        s = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                        
                        # 색상 채우기 (R, G, B, Alpha) -> Alpha=100 (0~255, 숫자가 클수록 불투명)
                        s.fill((*highlight_color_blue, 100)) 

                        for (hx, hy) in highlight_for_plan_coords:
                            # 좌표 유효성 검사 (맵 밖으로 나가는 것 방지)
                            if 0 <= hx < grid_width and 0 <= hy < grid_height:
                                # 그릴 위치 계산
                                draw_x = start_x + (hx * tile_w)
                                draw_y = map_start_y + (hy * tile_h)
                                
                                # 1) 반투명 색칠하기
                                window.blit(s, (draw_x, draw_y))
                                
                                # 2) 진한 테두리 그리기 (선택 사항)
                                rect = pygame.Rect(draw_x, draw_y, tile_w, tile_h)
                                pygame.draw.rect(window, highlight_color_blue, rect, 3) # 두께 3

    pygame.display.flip()


def generate_layout_dict(mdp):
    """
    레이아웃 정보를 문자열 대신 구조화된 딕셔너리로 반환합니다.
    """
    layout_data = {}

    # 출력할 이름 정의
    name_map = {
        "onion_dispenser": "Onion Dispenser",
        "dish_dispenser": "Dish Dispenser",
        "tomato_dispenser": "Tomato Dispenser",
        "serving": "Serving Loc",
        "pot": "Pot"
    }

    # 각 객체 타입별 순회
    for key, readable_name in name_map.items():
        # MDP에서 위치 정보 가져오기
        locations = getattr(mdp, f"get_{key}_locations")()
        
        # 해당 객체가 맵에 없으면 건너뜀
        if not locations:
            continue
            
        # 해당 타입의 리스트 초기화
        layout_data[key] = []

        for i, pos in enumerate(locations):
            # 딕셔너리 형태로 정보 저장
            item_info = {
                "id": i,
                "type": readable_name,            # 예: Onion Dispenser
                "full_name": f"<{readable_name} {i}>", # 예: <Onion Dispenser 0>
                "position": pos                   # 예: (0, 1)
            }
            layout_data[key].append(item_info)

    return layout_data

def main(variant):

    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']
    mode = variant['mode']
    render = variant['render'] 
    cook_time = variant['cook_time']
    visual_level = variant['visual_level']
    
    if VERSION == '1.1.0':
        mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS[layout])
    elif VERSION == '0.0.1':
        mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS[layout])

    layout_dict = generate_layout_dict(mdp)
    #print("layout_dict :",layout_dict)
    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()
    
    # 렌더링 옵션이 켜져 있을 때만 Pygame 초기화
    visualizer = None
    window_surface = None
    if render:
        pygame.init()
        visualizer = StateVisualizer(cook_time = cook_time)
        window_surface = pygame.display.set_mode((900, 600)) 
        pygame.display.set_caption("overcookedAI")
    
    p0_algo = variant['p0']
    p1_algo = variant['p1']
    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")

    start_time = time.time()
    results = []

    # [Rich Progress Bar 설정]
    with Progress(
        SpinnerColumn(),        
        TextColumn("[progress.description]{task.description}"), 
        BarColumn(),            
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
        TimeRemainingColumn(),  
        TextColumn(" | Mean Reward: [bold cyan]{task.fields[mean_score]}"), 
        transient=False         
    ) as progress:
        
        task_id = progress.add_task(f"[green]Simulating {episode} Episodes...", total=episode, mean_score="0")
        if variant['p0'] == 'Human' or variant['p1']== 'Human':
            render = True
            
        for i in range(episode):  
            agents_list = []
            for alg in [p0_algo, p1_algo]:
                if alg == "ProAgent" or alg == "EIRA" or alg == "EIRAAsync":
                    gpt_model = variant['gpt_model']
                    retrival_method = variant['retrival_method']
                    K = variant['K']
                    prompt_level = variant['prompt_level']
                    belief_revision = variant['belief_revision']
                    agent = make_agent(alg, mdp, layout, model=gpt_model, 
                                       prompt_level=prompt_level, 
                                       belief_revision=belief_revision, 
                                       retrival_method=retrival_method, K=K)
                elif alg == "Human":
                    agent = HumanAgent()
                elif alg == "BC":
                    agent = make_agent(alg, mdp, layout, seed_id=i) 
                else:
                    agent = make_agent(alg, mdp, layout)
                agents_list.append(agent)
            
            print("agents_list: ", *agents_list)
            team = AgentGroup(*agents_list)
            team.reset()
            env.reset()
            r_total = 0

            # =================================================================================
            # [WarmUp] 1. LLM 웜업 (Warm-up) & 첫 응답 대기 로직
            # =================================================================================
            if render and mode == 'exp':
                pro_agent_found = False
                for idx, agent in enumerate(agents_list):
                    # ProAgent 혹은 MyAgentAsync 등 LLM을 사용하는 에이전트인지 확인
                    if hasattr(agent, 'generate_ml_action'):
                        pro_agent_found = True
                        draw_centered_text(window_surface, 
                                           "Initializing AI Agent...", 
                                           "Waiting for the first thought...", 
                                           color=(0, 0, 255))
                        
                        print(f"\n[WarmUp] Triggering LLM for Agent {idx}...")
                        
                        # generate_ml_action 호출 (강제 생각)
                        _ = agent.generate_ml_action(env.state)
                        
                        print(f"[WarmUp] LLM Response Received!")
                
                # ProAgent가 있어서 응답을 받았으면 카운트다운 시작
                if pro_agent_found:
                    for count in range(3, 0, -1):
                        draw_centered_text(window_surface, 
                                           f"Game Starts in {count}...", 
                                           "AI is Ready!",
                                           color=(255, 0, 0))
                        time.sleep(1)
            # =================================================================================

            clock = pygame.time.Clock()
            if mode == 'exp':
                if (variant['p0'] == 'Human') or (variant['p1'] == 'Human'):
                    has_proagent = 1
                    if variant['p0'] == 'Human': num_human = 0
                    else: num_human = 1

                    if variant['p0'] in ['ProAgent', 'EIRA', 'EIRAAsync']: num_AI = 0
                    elif variant['p1'] in ['ProAgent', 'EIRA', 'EIRAAsync']: num_AI = 1
                    else:
                        has_proagent = 0
                        num_AI = -1
                    
                    for t in range(1, horizon+1):
                        # 1. 스텝 시작 시간 및 변수 초기화
                        step_start_time = pygame.time.get_ticks()
                        step_duration = 400  # 400ms 고정
                        
                        action_chosen = False 
                        chosen_action = (0, 0) # 기본값: 정지(Stay)

                        # 2. 렌더링 (첫 프레임)
                        first = True
                        if render and first:
                            first = False
                            current_thought = "No ProAgent"
                            if has_proagent == 1:
                                current_thought = agents_list[num_AI].current_thought 
                            
                            # 통합 렌더링 함수 호출
                            render_game(window=window_surface, visualizer=visualizer, env=env, step=t, horizon=horizon, reward=r_total, num_AI=num_AI,
                                        visual_level=visual_level, layout_dict=layout_dict, thought_msg=current_thought)

                        # 3. 400ms 대기 루프 (입력 처리 포함)
                        while True:
                            clock.tick(60)
                            
                            current_time = pygame.time.get_ticks()
                            elapsed = current_time - step_start_time

                            if elapsed >= step_duration:
                                break

                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    return
                                
                                if event.type == pygame.KEYDOWN and not action_chosen:
                                    key = event.key
                                    action = None
                                    if key == pygame.K_UP: action = (0, -1)
                                    elif key == pygame.K_DOWN: action = (0, 1)
                                    elif key == pygame.K_LEFT: action = (-1, 0)
                                    elif key == pygame.K_RIGHT: action = (1, 0)
                                    elif key == pygame.K_SPACE: action = "interact"
                                    
                                    if action is not None:
                                        chosen_action = action
                                        action_chosen = True 
                        
                        # 4. 행동 주입 및 환경 업데이트
                        agents_list[num_human].set_next_action(chosen_action)

                        s_t = env.state
                        a_t = []
                        a_t.append(team.agents[0].action(s_t))
                        
                        # 렌더링 업데이트 (생각 중 표시 가능)
                        if render:
                            current_thought = "No ProAgent"
                            if has_proagent == 1:
                                current_thought = agents_list[num_AI].current_thought
                            
                            render_game(window=window_surface, visualizer=visualizer, env=env, step=t, horizon=horizon, reward=r_total, num_AI=num_AI,
                                        visual_level=visual_level, layout_dict=layout_dict, thought_msg=current_thought)
                            pygame.event.pump()

                        a_t.append(team.agents[1].action(s_t))
                        a_t = tuple(a_t)
                        
                        obs, reward, done, env_info = env.step(a_t)
                        r_total += reward
                        
                        # 스텝 완료 후 렌더링 (Thinking 꺼짐)
                        if render:
                            render_game(window=window_surface, visualizer=visualizer, env=env, step=t, horizon=horizon, reward=r_total, num_AI=num_AI,
                                        visual_level=visual_level, layout_dict=layout_dict, thought_msg=current_thought)
                        
                        if done: break
                else:
                    # AI vs AI (or other modes)
                    for t in range(horizon):
                        clock.tick(3)
                        if render:
                            # AI 모드에서는 간단히 맵만 렌더링 (visual_level 0 효과)
                            render_game(window=window_surface, visualizer=visualizer, env=env, step=t, horizon=horizon, reward=r_total, num_pro=num_AI,
                                        visual_level=visual_level, layout_dict=layout_dict, thought_msg=current_thought)
                    
                        s_t = env.state
                        a_t = team.joint_action(s_t)
                        obs, reward, done, env_info = env.step(a_t)
                        r_total += reward
                        
                        pygame.event.pump() 

            elif mode == 'demo':
                pass
            
            results.append(r_total)
            current_mean = int(np.mean(results))
            progress.update(task_id, advance=1, mean_score=str(current_mean))

    # 전체 루프 종료 후
    end_time = time.time()
    print(f"\n\nTotal Cost time : {end_time - start_time:.3f}s-----\n")

    result_dict = {
        "input": variant,
        "raw_results": results,
        "mean_result": int(np.mean(results)),
        "std_result": float(np.std(results))
    }
    
    for (k,v) in result_dict.items():
        if k != "raw_results":
            print(f'{k}: {v}')

    if variant['save']:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if variant['log_dir'] == None:
            log_dir = f"experiments/{timestamp}_{layout}_{horizon}_{p0_algo}_{p1_algo}_{episode}numep"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        else:
            log_dir = variant['log_dir']

        print(f"Result saved to {log_dir}")
        safe_model_name = str(variant.get('gpt_model', 'model')).replace('/', '-')
        
        if p0_algo == "ProAgent"  or p1_algo == "ProAgent":
            json_file = f"{log_dir}/results_{episode}_{horizon}_{safe_model_name}_{variant['prompt_level']}.json"
        else:
            json_file = f"{log_dir}/results_{episode}_{horizon}.json"
            
        with open(json_file, "w") as f:
            json.dump(result_dict, f, indent=4)

    
if __name__ == '__main__':
    parser = ArgumentParser(description='OvercookedAI Experiment')

    # Basic Parsers
    parser.add_argument('--layout', '-l', type=str, default='cramped_room', choices=['cramped_room', 'asymmetric_advantages', 'coordination_ring', 'forced_coordination', 'counter_circuit'])
    parser.add_argument('--p0',  type=str, default='ProAgent', help='Algorithm for P0 agent 0')
    parser.add_argument('--p1', type=str, default='Greedy', help='Algorithm for P1 agent 1')
    parser.add_argument('--horizon', type=int, default=400, help='Horizon steps in one game')
    parser.add_argument('--cook_time', type=int, default=20)
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')
    parser.add_argument('--render', type=boolean_argument, default=True, help='Visualization on/off')
    parser.add_argument('--visual_level', type=int, default=0, help='0: map only, 1: emoji vis, 2: NL vis')

    # ProAgent Parsers
    parser.add_argument('--gpt_model', type=str, default='Qwen/Qwen3-VL-8B-Instruct', help='Model name')
    parser.add_argument('--api_base', type=str, default='http://localhost:8000/v1', help='Local LLM API URL')
    parser.add_argument('--api_key', type=str, default='EMPTY', help='API Key')
    parser.add_argument('--prompt_level', '-pl', type=str, default='l3-aip', choices=['l1-p', 'l2-ap', 'l3-aip'])
    parser.add_argument('--belief_revision', '-br', type=boolean_argument, default=False)
    parser.add_argument('--retrival_method', type=str, default="recent_k", choices=['recent_k', 'bert_topk'])
    parser.add_argument('--K', type=int, default=1)

    # Misc Parsers
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'demo'])                                
    parser.add_argument('--save', type=boolean_argument, default=True)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--debug', type=boolean_argument, default=True)

    args = parser.parse_args()
    variant = vars(args)

    main(variant)