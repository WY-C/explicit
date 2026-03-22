import sys
import os
import pygame
import argparse
import datetime
import json
import platform
import visualization_utils as vu 
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

try: from utils import make_agent
except ImportError: pass

TUTORIAL_GRID_DATA = ["XXXXPXXXX", "O       D", "X      2X", "X   1   X", "XXXXSXXXX"]

class OvercookedLogger:
    def __init__(self, log_dir="experiments/logs", variant=None):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        layout = variant.get('layout', 'unknown') if variant else 'unknown'
        cond_name = variant.get('name', 'tutorial') if variant else 'tutorial'
        self.filepath = os.path.join(self.log_dir, f"log_{timestamp}_{layout}_{cond_name}.jsonl")
        if variant: self._write_line({"metadata": variant})
    def log_step(self, step_data): self._write_line(step_data)
    def _write_line(self, data):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def run_tutorial(env, screen, visualizer, ai_agent=None, args=None):
    logger = OvercookedLogger(log_dir=args.log_dir, variant=vars(args))
    layout_dict = vu.generate_layout_dict(env.mdp)
    served, step, col_total = 0, 0, 0; current_thought = ""
    
    # 루프 진입 전 0번째 스텝 렌더링 (처음 시작 시 검은 화면/딜레이 방지)
    render_tutorial_game(screen, visualizer, env, step, served, args.visual_level, layout_dict, thought_msg=current_thought)
    
    while served < 2:
        # 에이전트 타임스텝 업데이트
        if ai_agent: ai_agent.current_timestep = step + 1
        
        human_action = Action.STAY; start_t = pygame.time.get_ticks(); act_chosen = False
        if not args.is_async:
            render_tutorial_game(screen, visualizer, env, step, served, args.visual_level, layout_dict, thought_msg=current_thought)
            while not act_chosen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP: human_action = Direction.NORTH; act_chosen = True
                        elif event.key == pygame.K_DOWN: human_action = Direction.SOUTH; act_chosen = True
                        elif event.key == pygame.K_LEFT: human_action = Direction.WEST; act_chosen = True
                        elif event.key == pygame.K_RIGHT: human_action = Direction.EAST; act_chosen = True
                        elif event.key == pygame.K_SPACE: human_action = Action.INTERACT; act_chosen = True
                pygame.time.delay(10)
        else:
            while pygame.time.get_ticks() - start_t < args.delay:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                    if event.type == pygame.KEYDOWN and not act_chosen:
                        if event.key == pygame.K_UP: human_action = Direction.NORTH; act_chosen = True
                        elif event.key == pygame.K_DOWN: human_action = Direction.SOUTH; act_chosen = True
                        elif event.key == pygame.K_LEFT: human_action = Direction.WEST; act_chosen = True
                        elif event.key == pygame.K_RIGHT: human_action = Direction.EAST; act_chosen = True
                        elif event.key == pygame.K_SPACE: human_action = Action.INTERACT; act_chosen = True
                pygame.time.delay(1)

        ai_action = ai_agent.action(env.state, partner_action=human_action) if ai_agent else Action.STAY
        current_thought = getattr(ai_agent, 'current_thought', "") if ai_agent else ""
        
        # 상호 충돌 감지
        old_p0, old_p1 = env.state.players[0].position, env.state.players[1].position
        int_p0 = (old_p0[0] + human_action[0], old_p0[1] + human_action[1]) if human_action in Direction.ALL_DIRECTIONS else old_p0
        int_p1 = (old_p1[0] + ai_action[0], old_p1[1] + ai_action[1]) if ai_action in Direction.ALL_DIRECTIONS else old_p1
        is_col = (int_p0 == int_p1 and (human_action != Action.STAY or ai_action != Action.STAY)) or \
                 (int_p0 == old_p1 and int_p1 == old_p0 and human_action != Action.STAY and ai_action != Action.STAY) or \
                 (human_action in Direction.ALL_DIRECTIONS and int_p0 == old_p1 and ai_action not in Direction.ALL_DIRECTIONS) or \
                 (ai_action in Direction.ALL_DIRECTIONS and int_p1 == old_p0 and human_action not in Direction.ALL_DIRECTIONS)
        if is_col: col_total += 1
        
        try:
            _, reward, _, _ = env.step((human_action, ai_action))
            if reward > 0: served += 1
        
        except ValueError as e:
            print(f"[Tutorial Env Error] Step failed: {e}. Attempting recovery with STAY.")
            _, reward, _, _ = env.step((human_action, Action.STAY))
            ai_agent.generate_ml_action(env.state)
        
        if reward > 0: served += 1
        step += 1
        logger.log_step({
            "timestep": step, 
            "joint_action": [human_action, ai_action], 
            "inter_collision": is_col, 
            "reward": reward, 
            "cumulative_reward": served * 20
        })
        if args.is_async:
            render_tutorial_game(screen, visualizer, env, step, served, args.visual_level, layout_dict, thought_msg=current_thought)
            
    return served, step, col_total

def render_tutorial_game(window, visualizer, env, step, served_count, visual_level, layout_dict, thought_msg=None):
    # 💡 [핵심 해결책] vu.render_game 내부의 강제 화면 업데이트 무력화 (Monkey Patching)
    # 이걸 통해 vu.render_game이 화면을 먼저 송출해버려서 깜빡이는 현상을 완벽 차단합니다.
    _original_flip = pygame.display.flip
    _original_update = pygame.display.update
    pygame.display.flip = lambda: None
    pygame.display.update = lambda *args, **kwargs: None
    
    try:
        # 마지막 인자 True는 AI intention 표시 옵션이므로 그대로 유지
        vu.render_game(window, visualizer, env, step, 40, served_count * 20, 1, visual_level, layout_dict, thought_msg, True)
    finally:
        # 게임 렌더링이 끝나면 업데이트 함수를 원상 복구
        pygame.display.flip = _original_flip
        pygame.display.update = _original_update
    
    # 영문 텍스트에 맞게 폰트 설정
    f_title = pygame.font.SysFont("arial", 20, bold=True)
    f_small = pygame.font.SysFont("arial", 14)
    f_step = pygame.font.SysFont("arial", 15, bold=True)
    
    # 왼쪽 아래 가이드 박스 (조작 및 튜토리얼 미션)
    guide_x, guide_y = 20, 470
    pygame.draw.rect(window, (252, 252, 252), (guide_x, guide_y, 280, 260), border_radius=12)
    pygame.draw.rect(window, (60, 60, 60), (guide_x, guide_y, 280, 260), 2, border_radius=12)
    
    window.blit(f_title.render("Game Controls", True, (0, 0, 0)), (guide_x + 15, guide_y + 15))
    pygame.draw.line(window, (200, 200, 200), (guide_x + 15, guide_y + 45), (guide_x + 265, guide_y + 45), 1)
    window.blit(f_step.render("Move: Arrow Keys", True, (50, 50, 50)), (guide_x + 15, guide_y + 55))
    window.blit(f_step.render("Interact: Spacebar", True, (50, 50, 50)), (guide_x + 15, guide_y + 80))
    
    window.blit(f_title.render("Tutorial Mission", True, (0, 0, 0)), (guide_x + 15, guide_y + 120))
    mission_steps = [
        "1. Put 3 onions into the pot.", 
        "2. Wait until the soup is cooked.", 
        "3. Grab a dish and plate the soup.", 
        "4. Deliver the soup to the grey counter."
    ]
    for i, txt in enumerate(mission_steps): 
        window.blit(f_small.render(txt, True, (80, 80, 80)), (guide_x + 15, guide_y + 150 + (i * 25)))

    # 오른쪽 아래 가이드 박스 (AI 시각화 안내)
    if visual_level in [1, 2]:
        ai_guide_x, ai_guide_y = 600, 470
        pygame.draw.rect(window, (252, 252, 252), (ai_guide_x, ai_guide_y, 280, 260), border_radius=12)
        pygame.draw.rect(window, (60, 60, 60), (ai_guide_x, ai_guide_y, 280, 260), 2, border_radius=12)
        
        if visual_level == 2:
            window.blit(f_title.render("AI Highlight Guide", True, (0, 0, 0)), (ai_guide_x + 15, ai_guide_y + 15))
            pygame.draw.line(window, (200, 200, 200), (ai_guide_x + 15, ai_guide_y + 45), (ai_guide_x + 265, ai_guide_y + 45), 1)
            window.blit(f_step.render("White: AI's plan", True, (100, 100, 100)), (ai_guide_x + 15, ai_guide_y + 55))
            window.blit(f_step.render("Blue: Prediction of your plan", True, (30, 30, 150)), (ai_guide_x + 15, ai_guide_y + 80))
        
        elif visual_level == 1:
            window.blit(f_title.render("AI Bubble Guide", True, (0, 0, 0)), (ai_guide_x + 15, ai_guide_y + 15))
            pygame.draw.line(window, (200, 200, 200), (ai_guide_x + 15, ai_guide_y + 45), (ai_guide_x + 265, ai_guide_y + 45), 1)
            window.blit(f_step.render("Top: Prediction of your plan", True, (50, 50, 50)), (ai_guide_x + 15, ai_guide_y + 55))
            window.blit(f_step.render("Bottom: AI's plan", True, (50, 50, 50)), (ai_guide_x + 15, ai_guide_y + 80))
            
        lines = [
            "(* Depending on the condition,",
            "only the AI's plan may be shown,",
            "or both the plan and prediction.)"
        ]

        for i, line in enumerate(lines):
            text = f_small.render(line, True, (100, 100, 100))
            window.blit(text, (ai_guide_x + 15, ai_guide_y + 120 + i * 20))

    # 모든 요소를 온전히 다 그린 후, 여기서 최종적으로 단 한 번만 화면 업데이트!
    pygame.display.flip()