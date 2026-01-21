"""
todo
=연구=
마일스톤 (계획, 오버뷰, 청사진) 만들기

파랑머리가 이야기하는 것처럼
파랑머리: 의도, 계획

이동 경로를 모두 저장했다가 보여주기
-> 이전 3개 정도만 보여주기
프롬프트 예시 추가하기


400ms이 fixed인지 참가자의 평균인지

가장 가까운 곳이 아니라, LLM이 어떤 pot에 넣을지까지 정하기

첫 스텝 답장이 온 다음에 게임 시작하기.


=벤치마크=
내 환경의 ProAgent와 비교

=논문=
ProAgent가 문단의 첫 문장에 나올 것이 아니라, detail에 조금만 들어가야 한다.


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

import pygame
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

print(f'\n----This overcook version is {VERSION}----\n')

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action

from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent

from overcooked_ai_py.agents.agent import Agent

# [수정됨] HumanAgent: set_next_action 구조 사용
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


def render_game_emoji(window, visualizer, env, step, horizon, reward, num_pro, thought_msg=None, is_thinking=False):
    if not window or not visualizer:
        return

    # 1. 배경 초기화
    window.fill((255, 255, 255)) 
    screen_width, screen_height = window.get_size()

    # 2. 상단 상태 정보 (맨 위 고정)
    font = pygame.font.SysFont("malgungothic", 30) 
    text = font.render(f"Step: {step}/{horizon} | Reward: {reward}", True, (0, 0, 0))
    window.blit(text, (10, 10))

    # -----------------------------------------------------------
    # 3. 텍스트 그리기 (먼저 배치하여 높이 계산)
    # -----------------------------------------------------------
    
    # 텍스트가 시작될 기본 Y 위치
    current_y = 60  
    
    if thought_msg:
        # 폰트 및 설정
        bubble_font = pygame.font.SysFont("malgungothic", 28) 
        text_color = (0, 0, 0)
        
        icon_size = 45    
        line_height = 45
        left_margin = 10  

        # 아이콘 매핑
        icon_map = {
            "fill_dish_with_soup": "soup-onion-cooked",
            "wait": "stay",
            "deliver_soup": "EAST-soup-onion",
            "onion": "onions", "양파": "onions",
            "dish": "dishes", "접시": "dishes",
            "pot": "pot", "냄비": "pot",
            "soup": "soup", "수프": "soup"
        }

        def chef_frame_name(d, obj): return f"{d}-{obj}" if obj else d
        def hat_frame_name(d, color): return f"{d}-{color}hat"

        # [Step 1] 셰프 아이콘 및 콜론(:) 그리기
        chef_surf = pygame.Surface((visualizer.UNSCALED_TILE_SIZE, visualizer.UNSCALED_TILE_SIZE), pygame.SRCALPHA)
        visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), chef_frame_name("SOUTH", None))
        if num_pro == 0:
            visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), hat_frame_name("SOUTH", "blue"))
        else:
            visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), hat_frame_name("SOUTH", "green"))
        
        
        scaled_chef = pygame.transform.scale(chef_surf, (icon_size, icon_size))
        window.blit(scaled_chef, (left_margin, current_y)) 
        
        colon_surf = bubble_font.render(" : ", True, text_color)
        colon_x = left_margin + icon_size 
        colon_y_colon = current_y + (line_height - colon_surf.get_height()) // 2
        window.blit(colon_surf, (colon_x, colon_y_colon))

        text_start_x = colon_x + colon_surf.get_width()

        # [Step 2] 텍스트 파싱 및 줄별 그리기
        raw_lines = thought_msg.split('\n')
        
        for line in raw_lines:
            line = line.strip()
            if not line: continue 

            display_text = ""
            if "Intention" in line: display_text = "You want"
            elif "Plan" in line: display_text = "I will do"
            else: continue 

            line_lower = line.lower()
            sprite_name = None
            for key, s_name in icon_map.items():
                if key in line_lower:
                    sprite_name = s_name
                    break 

            current_x = text_start_x
            text_surf = bubble_font.render(display_text, True, text_color)
            text_y_pos = current_y + (line_height - text_surf.get_height()) // 2
            window.blit(text_surf, (current_x, text_y_pos))
            
            current_x += text_surf.get_width() + 10 

            if sprite_name:
                # 1. 그릴 표면 생성 및 검은색 배경 채우기 (이전 요청사항 반영)
                obj_surf = pygame.Surface((visualizer.UNSCALED_TILE_SIZE, visualizer.UNSCALED_TILE_SIZE), pygame.SRCALPHA)
                #obj_surf.fill((0, 0, 0)) # 검은 배경
                
                # ---------------------------------------------------------
                # [수정됨] 'stay'인 경우 JSON 조회 대신 파일 직접 로드
                # ---------------------------------------------------------
                if sprite_name == 'stay':
                    try:
                        # 같은 경로에 있는 'stay.png' 불러오기
                        raw_img = pygame.image.load('stay.png').convert_alpha()
                        
                        # 불러온 이미지를 타일 크기(UNSCALED_TILE_SIZE)에 맞게 조정
                        # (원본 이미지 크기가 타일 크기와 다를 수 있으므로 안전장치)
                        resized_img = pygame.transform.scale(raw_img, (visualizer.UNSCALED_TILE_SIZE, visualizer.UNSCALED_TILE_SIZE))
                        
                        # 검은 배경 위에 그리기
                        obj_surf.blit(resized_img, (0, 0))
                    except FileNotFoundError:
                        print("Error: 'stay.png' file not found.")
                
                # 'stay'가 아닌 경우 기존 로직 수행 (JSON 사용)
                else:
                    try: visualizer.TERRAINS_IMG.blit_on_surface(obj_surf, (0, 0), sprite_name)
                    except KeyError:
                        try: visualizer.OBJECTS_IMG.blit_on_surface(obj_surf, (0, 0), sprite_name)
                        except KeyError:
                            try: visualizer.CHEFS_IMG.blit_on_surface(obj_surf, (0, 0), sprite_name)
                            except: pass

                # 화면에 최종 출력 (아이콘 크기로 조정)
                scaled_obj = pygame.transform.scale(obj_surf, (icon_size, icon_size))
                window.blit(scaled_obj, (current_x, current_y + (line_height - icon_size)//2))

            # 줄바꿈: Y 위치 증가
            current_y += line_height

    # -----------------------------------------------------------
    # 4. 게임 화면 그리기 (텍스트 바로 아래에 배치)
    # -----------------------------------------------------------
    
    # [핵심] 텍스트가 끝난 위치(current_y)에 약간의 여백(20)을 두고 맵을 그립니다.
    # 만약 텍스트가 없으면 current_y는 초기값 60이므로 상단에 그려집니다.
    #print("current_y: ", current_y)
    map_start_y = 170

    state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
    surf_width, surf_height = state_surface.get_size()
    
    start_x = (screen_width - surf_width) // 2
    
    window.blit(state_surface, (start_x, map_start_y))

    pygame.display.flip()

def render_game_NL(window, visualizer, env, step, horizon, reward, num_pro, thought_msg=None, is_thinking=False):
    """
    자연어 텍스트 앞에 셰프 아이콘을 표시하여 
    셰프가 말하는 것처럼 연출하는 함수
    """
    if not window or not visualizer:
        return

    # 1. 화면 초기화
    window.fill((255, 255, 255))
    screen_width, screen_height = window.get_size()

    # 폰트 및 설정
    font_name = "malgungothic" if "malgungothic" in pygame.font.get_fonts() else None
    font_header = pygame.font.SysFont(font_name, 30)
    font_text = pygame.font.SysFont(font_name, 24)
    
    icon_size = 45    # 아이콘 크기
    line_height = 40  # 줄 간격
    left_margin = 10  # 왼쪽 여백
    current_y = 60    # 텍스트/아이콘 시작 Y 위치

    # -----------------------------------------------------------
    # 2. 상단 정보 (Step, Reward)
    # -----------------------------------------------------------
    header_text = font_header.render(f"Step: {step}/{horizon} | Reward: {reward}", True, (0, 0, 0))
    window.blit(header_text, (10, 10))

    # -----------------------------------------------------------
    # 3. 셰프 아이콘 + 자연어 대화창 그리기
    # -----------------------------------------------------------
    if thought_msg:
        # --- [A] 셰프 이미지 생성 (render_game_emoji에서 가져옴) ---
        def chef_frame_name(d, obj): return f"{d}-{obj}" if obj else d
        def hat_frame_name(d, color): return f"{d}-{color}hat"

        # 투명 배경의 서피스 생성
        chef_surf = pygame.Surface((visualizer.UNSCALED_TILE_SIZE, visualizer.UNSCALED_TILE_SIZE), pygame.SRCALPHA)
        
        # 1. 셰프 몸통 그리기
        visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), chef_frame_name("SOUTH", None))
        
        # 2. 모자 그리기 (Player 0: 파랑 / Player 1: 초록)
        # num_pro가 0이면 Blue, 아니면 Green (필요에 따라 반대로 설정 가능)
        if num_pro == 0:
            visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), hat_frame_name("SOUTH", "blue"))
        else:
            visualizer.CHEFS_IMG.blit_on_surface(chef_surf, (0, 0), hat_frame_name("SOUTH", "green"))
        
        # 3. 크기 조절 후 화면에 배치
        scaled_chef = pygame.transform.scale(chef_surf, (icon_size, icon_size))
        window.blit(scaled_chef, (left_margin, current_y))
        
        # 4. 콜론 " : " 그리기
        colon_surf = font_header.render(" : ", True, (0, 0, 0))
        colon_x = left_margin + icon_size
        colon_y = current_y + (icon_size - colon_surf.get_height()) // 2
        window.blit(colon_surf, (colon_x, colon_y))
        
        # 텍스트 시작 X 좌표 (아이콘 + 콜론 뒤)
        text_start_x = colon_x + colon_surf.get_width()

        # --- [B] 자연어 텍스트 출력 ---
        
        # 텍스트 다듬기 (필요한 부분만 자르기)
        if "Intention for Player" in thought_msg:
            thought_msg = thought_msg[thought_msg.find("Intention for Player"):]
            thought_msg = thought_msg.replace("Intention for Player 1", "Intention for Partner")
        if "Plan for Player 0" in thought_msg:
            thought_msg = thought_msg.replace("Plan for Player 0", "\nPlan for Me")

        lines = thought_msg.split('\n')
        
        # 첫 번째 줄은 아이콘 높이에 맞춰 출력
        # 그 다음 줄부터는 아이콘 아래쪽으로 줄바꿈
        text_y_pos = current_y + (icon_size - font_text.get_height()) // 2 # 수직 중앙 정렬
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            text_surf = font_text.render(line, True, (0, 0, 0))
            
            # 첫 줄은 아이콘 옆에, 그 다음 줄부터는 들여쓰기 맞춰서 출력
            draw_x = text_start_x if i == 0 else text_start_x 
            # (만약 두번째 줄부터 아이콘 밑으로 보내고 싶으면 draw_x = left_margin 으로 변경)
            
            window.blit(text_surf, (draw_x, text_y_pos))
            
            # 다음 줄로 이동
            text_y_pos += line_height
            
        # 텍스트가 끝난 최종 Y 위치 업데이트 (맵 그릴 위치 계산용)
        current_y = text_y_pos

    # -----------------------------------------------------------
    # 4. 게임 맵(State) 그리기
    # -----------------------------------------------------------
    # 텍스트와 겹치지 않게 아래쪽에 배치 (최소 Y값 170 보장)
    map_start_y = max(current_y + 20, 170)

    state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
    surf_width, surf_height = state_surface.get_size()
    
    start_x = (screen_width - surf_width) // 2
    
    window.blit(state_surface, (start_x, map_start_y))

    # -----------------------------------------------------------
    # 5. 생각 중 (Thinking) 표시
    # -----------------------------------------------------------
    if is_thinking:
        loading_font = pygame.font.SysFont(font_name, 40, bold=True)
        loading_text = loading_font.render("Agent is thinking...", True, (255, 0, 0))
        
        text_rect = loading_text.get_rect(center=(screen_width // 2, 80))
        bg_rect = text_rect.inflate(20, 10)
        
        pygame.draw.rect(window, (255, 255, 255), bg_rect)
        pygame.draw.rect(window, (0, 0, 0), bg_rect, 2)
        window.blit(loading_text, text_rect)

    pygame.display.flip()


def main(variant):
    #step_duration = 500
    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']
    mode = variant['mode']
    render = variant['render'] # 렌더링 여부 확인
    cook_time = variant['cook_time']
    visual_level = variant['visual_level']
    
    if VERSION == '1.1.0':
        mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS[layout])
    elif VERSION == '0.0.1':
        mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS[layout])

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()
    
    # 렌더링 옵션이 켜져 있을 때만 Pygame 초기화
    visualizer = None
    window_surface = None
    if render:
        pygame.init()
        visualizer = StateVisualizer(cook_time = cook_time)
        window_surface = pygame.display.set_mode((900, 600)) 
        pygame.display.set_caption("ProAgent vs Human")
    
    p0_algo = variant['p0']
    p1_algo = variant['p1']
    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")

    start_time = time.time()
    results = []

    # [Rich Progress Bar 설정]
    with Progress(
        SpinnerColumn(),        # 로딩 스피너
        TextColumn("[progress.description]{task.description}"), # 설명 텍스트
        BarColumn(),            # 진행 바
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), # 퍼센트
        TimeRemainingColumn(),  # 남은 시간
        TextColumn(" | Mean Reward: [bold cyan]{task.fields[mean_score]}"), # 커스텀 필드 (평균 점수)
        transient=False         # 완료 후에도 바를 사라지게 하지 않음
    ) as progress:
        
        task_id = progress.add_task(f"[green]Simulating {episode} Episodes...", total=episode, mean_score="0")
        if variant['p0'] == 'Human' or variant['p1']== 'Human':
            render = True
            
        for i in range(episode):  
            agents_list = []
            for alg in [p0_algo, p1_algo]:
                if alg == "ProAgent" or alg == "MyAgent" or alg == "MyAgentAsync":
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
                    agent = make_agent(alg, mdp, layout, seed_id=i) # 매 에피소드마다 시드 변경
                else:
                    agent = make_agent(alg, mdp, layout)
                agents_list.append(agent)
            print("agents_list: ", *agents_list)
            team = AgentGroup(*agents_list)
            team.reset()
            env.reset()
            r_total = 0

            clock = pygame.time.Clock()
            if mode == 'exp':
                if (variant['p0'] == 'Human') or (variant['p1'] == 'Human'):
                    has_proagent = 1
                    if variant['p0'] == 'Human':
                        num_human = 0
                    else:
                        num_human = 1
                    if variant['p0'] == 'ProAgent' or variant['p0'] == 'MyAgent' or variant['p0'] == 'MyAgentAsync':
                        num_pro = 0
                    elif variant['p1'] == 'ProAgent'or variant['p1'] == 'MyAgent' or variant['p1'] == 'MyAgentAsync':
                        num_pro = 1
                    else:
                        has_proagent = 0
                    
                    for t in range(horizon):
                        # 1. 스텝 시작 시간 및 변수 초기화
                        step_start_time = pygame.time.get_ticks()
                        step_duration = 400  # 400ms 고정
                        
                        # [중요] event.clear() 제거! (입력 씹힘 방지)
                        # 대신, 이번 턴에 행동을 정했는지 확인하는 플래그 사용
                        action_chosen = False 
                        chosen_action = (0, 0) # 기본값: 정지(Stay)

                        # 2. 렌더링 (첫 프레임)
                        first = True
                        if render and first:
                            first = False
                            if has_proagent == 1:
                                current_thought = agents_list[num_pro].current_thought 
                            else:
                                current_thought = "No ProAgent"
                            if visual_level == 0:
                                render_game_emoji(window_surface, visualizer, env, t, horizon, r_total, num_pro, current_thought, is_thinking=False)
                            elif visual_level == 1:
                                render_game_emoji(window_surface, visualizer, env, t, horizon, r_total, num_pro, current_thought, is_thinking=False)
                        
                            else:
                                render_game_NL(window_surface, visualizer, env, t, horizon, r_total, num_pro, current_thought, is_thinking=False)
                        

                            
                        # 3. [핵심] 400ms가 지날 때까지 무조건 루프를 돕니다.
                        # 입력을 빨리 했어도 여기서 시간을 떼웁니다 (Non-blocking Wait).
                        while True:
                            # 60FPS 유지
                            #todo: 입력 버퍼 비우기
                            clock.tick(100)
                            
                            current_time = pygame.time.get_ticks()
                            elapsed = current_time - step_start_time

                            # 400ms가 지나면 루프 탈출 -> 다음 스텝 진행
                            if elapsed >= step_duration:
                                break

                            # 이벤트 처리
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    return
                                
                                # 키 입력 처리 (이미 이번 턴에 행동을 결정했다면 무시)
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
                                        # [중요] break 하지 않음! 
                                        # 행동은 입력받았지만, 시간(400ms)은 다 채우고 나갑니다.
                        
                        # 4. 루프가 끝나면(400ms 경과) 결정된 행동을 에이전트에 주입
                        # 입력이 없었으면 기본값 (0,0)이 들어감
                        agents_list[num_human].set_next_action(chosen_action)

                        # 5. 환경 업데이트 (Step)
                        s_t = env.state
                        a_t = []
                        a_t.append(team.agents[0].action(s_t))
                        
                        if render:
                            if has_proagent == 1:
                                current_thought = agents_list[num_pro].current_thought
                            if visual_level == 0:
                                render_game_emoji(window_surface, visualizer, env, t, horizon, r_total, num_pro, current_thought, is_thinking=True)
                            elif visual_level == 1:
                                render_game_emoji(window_surface, visualizer, env, t, horizon, r_total, num_pro, current_thought, is_thinking=True)
                        
                            else:
                                render_game_NL(window_surface, visualizer, env, t, horizon, r_total, num_pro, current_thought, is_thinking=True)
                            pygame.event.pump()

                        a_t.append(team.agents[1].action(s_t))
                        a_t = tuple(a_t)
                        
                        obs, reward, done, env_info = env.step(a_t)
                        r_total += reward
                        if visual_level == 0:
                            render_game_emoji(window_surface, visualizer, env, t, horizon, r_total, num_pro, current_thought, is_thinking=False)
                        elif visual_level == 1:
                            render_game_emoji(window_surface, visualizer, env, t, horizon, r_total, num_pro, current_thought, is_thinking=False)
                    
                        else:
                            render_game_NL(window_surface, visualizer, env, t, horizon, r_total, num_pro, current_thought, is_thinking=False)
                        
                        # [삭제] 마지막의 pygame.time.wait()는 이제 필요 없습니다.
                        # 위쪽의 while 루프가 정확히 시간을 맞춰줍니다.
                        
                        if done: break
                else:
                    
                    for t in range(horizon):
                        clock.tick(3)
                        # [렌더링 로직] render=True일 때만 실행
                        if render:
                            state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
                            
                            screen_width, screen_height = window_surface.get_size()
                            surf_width, surf_height = state_surface.get_size()
                            start_x = (screen_width - surf_width) // 2
                            start_y = (screen_height - surf_height) // 2
                            
                            window_surface.fill((255, 255, 255)) 
                            window_surface.blit(state_surface, (start_x, start_y))
                            
                            font = pygame.font.SysFont(None, 36)
                            text = font.render(f"Step: {t}/{horizon} | Reward: {r_total}", True, (0, 0, 0))
                            window_surface.blit(text, (10, 10))        
                    
                        s_t = env.state
                        
                        # 에이전트 행동 결정
                        a_t = team.joint_action(s_t)

                        #print("a_t:", a_t)
                        obs, reward, done, env_info = env.step(a_t)
                        pygame.display.flip()
                        pygame.event.pump() 
                        r_total += reward

            elif mode == 'demo':
                pass
            
            # 결과 저장 및 진행 바 업데이트
            results.append(r_total)
            current_mean = int(np.mean(results))
            
            # 진행 바 한 칸 전진 및 평균 점수 텍스트 갱신
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
    
    # 결과 요약 출력
    for (k,v) in result_dict.items():
        if k != "raw_results":
            print(f'{k}: {v}')

    # 파일 저장
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
    parser.add_argument('--visual_level', type=int, default=0, help='0: no vis, 1: emoji vis, 2: NL vis')

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