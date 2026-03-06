import sys
import pygame
import re
import argparse
import threading
from distutils.util import strtobool

# 오버쿡드 환경 및 에이전트 임포트
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# 커스텀 모듈 연결 (사용자의 프로젝트 구조에 맞춤)
try:
    from utils import make_agent
except ImportError:
    pass

# =========================================================
# 1. Utility Functions (Rendering & Parsing)
# =========================================================

def boolean_argument(value):
    if isinstance(value, bool): return value
    return bool(strtobool(value))

def render_rich_text(text, font_normal, font_bold, color=(0, 0, 0)):
    parts = text.split('**')
    rendered_parts = []
    total_width = 0
    max_height = 0

    for i, part in enumerate(parts):
        if not part: continue
        is_bold = (i % 2 != 0)
        font = font_bold if is_bold else font_normal
        surf = font.render(part, True, color)
        rendered_parts.append(surf)
        total_width += surf.get_width()
        max_height = max(max_height, surf.get_height())

    if total_width == 0:
        return font_normal.render(text, True, color)

    final_surf = pygame.Surface((total_width, max_height), pygame.SRCALPHA)
    current_x = 0
    for surf in rendered_parts:
        # 각 파트를 Y축 중앙 정렬하여 크기 차이 보정
        y_pos = (max_height - surf.get_height()) // 2
        final_surf.blit(surf, (current_x, y_pos))
        current_x += surf.get_width()

    return final_surf

def transform_to_english_natural(skill_name, idx, is_thought, target_objects):
    direction_str = ""
    if target_objects and len(target_objects) >= 1:
        def get_pos(obj): return obj['position'] if isinstance(obj, dict) else obj
        t_pos = get_pos(target_objects[idx if idx < len(target_objects) else 0])
        direction_str = "left " if t_pos[0] < 4 else "right "

    skill_map = {
        "pickup_onion": f"I'll grab the **{direction_str}onion**..." if not is_thought else f"Seems like he's going for the **{direction_str}onion**...",
        "pickup_dish": f"I'll get the **{direction_str}dish**..." if not is_thought else f"Seems like he's getting the **{direction_str}dish**...",
        "put_onion_in_pot": f"I'll put this in the **{direction_str}pot**..." if not is_thought else f"Seems like he's putting it in the **{direction_str}pot**...",
        "fill_dish_with_soup": f"I'll plate the **{direction_str}soup**..." if not is_thought else f"Seems like he's plating the **{direction_str}soup**...",
        "deliver_soup": "I'll deliver the **soup**..." if not is_thought else "Seems like he's delivering the **soup**...",
        "wait": "I'll **wait** for a sec..." if not is_thought else "Seems like he's **waiting**..."
    }
    return skill_map.get(skill_name, "Thinking...")

def generate_layout_dict(mdp):
    layout_data = {}
    mapping = {"onion_dispenser": "Onion", "dish_dispenser": "Dish", "pot": "Pot", "serving": "Serve"}
    for key, _ in mapping.items():
        if hasattr(mdp, f"get_{key}_locations"):
            locs = getattr(mdp, f"get_{key}_locations")()
            layout_data[key] = [{"id": i, "position": pos} for i, pos in enumerate(locs)]
    return layout_data

def parse_separate_highlights(thought_text, layout_dict, num_AI=None):
    if not hasattr(parse_separate_highlights, "cached_inf"): parse_separate_highlights.cached_inf = []
    if not hasattr(parse_separate_highlights, "cached_plan"): parse_separate_highlights.cached_plan = []
    
    def _to_coords(text):
        if not text: return []
        k_map = {"onion": "onion_dispenser", "dish": "dish_dispenser", "pot": "pot", "soup": "pot", "serve": "serving"}
        idx_match = re.search(r'(\d+)', text)
        idx = int(idx_match.group(1)) if idx_match else 0
        for k, v in k_map.items():
            if k in text.lower() and v in layout_dict and idx < len(layout_dict[v]):
                return [layout_dict[v][idx]['position']]
        return []

    p_m = re.search(r'Plan\s+for\s+Player\s+\d+:\s*"([^"]+)"', thought_text, re.IGNORECASE)
    if p_m: 
        coords = _to_coords(p_m.group(1))
        if coords: parse_separate_highlights.cached_plan = coords
    
    i_m = re.search(r'Intention\s+for\s+Player\s+\d+:\s*"([^"]+)"', thought_text, re.IGNORECASE)
    if i_m:
        coords = _to_coords(i_m.group(1))
        if coords: parse_separate_highlights.cached_inf = coords

    return parse_separate_highlights.cached_inf, parse_separate_highlights.cached_plan

def get_player_screen_pos(p_idx, env, map_y, start_x, s_w, s_h):
    pos = env.state.players[p_idx].position
    gw, gh = len(env.mdp.terrain_mtx[0]), len(env.mdp.terrain_mtx)
    tw, th = s_w / gw, s_h / gh
    return start_x + (pos[0] * tw) + (tw / 2), map_y + (pos[1] * th) + (th / 2)

# ... (기존 임포트 및 유틸리티 함수들은 동일하므로 생략) ...

def render_game(window, visualizer, env, step, horizon, reward, num_AI, visual_level, layout_dict,
                thought_msg=None, show_intention=True):
    """
    오버쿡드 게임 화면 및 튜토리얼 UI 렌더링 함수
    """
    if not window or not visualizer:
        return

    # 1. 배경 및 기본 맵 렌더링
    window.fill((255, 255, 255)) 
    screen_width, screen_height = window.get_size()
    
    # 상단 헤더 정보
    f_header = pygame.font.SysFont("arial", 26, bold=True)
    header_surf = f_header.render(f"Step: {step}/{horizon} | Reward: {reward}", True, (40, 40, 40))
    window.blit(header_surf, (25, 20))

    # 맵 렌더링 위치 계산
    map_start_y = 100
    state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
    surf_width, surf_height = state_surface.get_size()
    start_x = (screen_width - surf_width) // 2
    window.blit(state_surface, (start_x, map_start_y))

    # 그리드 좌표 변수 정의 (에러 방지용 필수 변수)
    grid_width = len(env.mdp.terrain_mtx[0])
    grid_height = len(env.mdp.terrain_mtx)
    tile_w = surf_width / grid_width
    tile_h = surf_height / grid_height
    
    # 공통 사용 폰트
    f_title = pygame.font.SysFont("arial", 18, bold=True)
    f_small = pygame.font.SysFont("arial", 14)
    f_step = pygame.font.SysFont("arial", 15, bold=True)
    f_desc = pygame.font.SysFont("arial", 12, italic=True)

    # ==========================================
    # 2. [Tutorial Guide] 좌측 가이드 패널 (무조건 출력)
    # ==========================================
    guide_x, guide_y = 0, 470
    # 배경 박스
    pygame.draw.rect(window, (252, 252, 252), (guide_x, guide_y, 250, 280), border_radius=12)
    pygame.draw.rect(window, (60, 60, 60), (guide_x, guide_y, 250, 280), 2, border_radius=12)

    window.blit(f_title.render("GAME GUIDE", True, (0, 0, 0)), (guide_x + 15, guide_y + 15))
    
    # 조작법
    pygame.draw.line(window, (200, 200, 200), (guide_x + 15, guide_y + 40), (guide_x + 235, guide_y + 40), 1)
    window.blit(f_step.render("Move: Arrow Keys", True, (50, 50, 50)), (guide_x + 15, guide_y + 50))
    window.blit(f_step.render("Action: Space Bar (Pick/Put)", True, (50, 50, 50)), (guide_x + 15, guide_y + 70))

    # 요리 미션 안내
    window.blit(f_title.render("MISSION STEPS", True, (0, 0, 0)), (guide_x + 15, guide_y + 110))
    mission_steps = [
        "1. Put 3 Onions in the Pot",
        "2. Wait until it finishes cooking",
        "3. Grab a Dish for the soup",
        "4. Deliver soup to Serving Loc"
    ]
    for i, txt in enumerate(mission_steps):
        window.blit(f_step.render(txt, True, (80, 80, 80)), (guide_x + 15, guide_y + 140 + (i * 30)))

    # ==========================================
    # 3. [Visual Level 1] 자연어 말풍선 (Stacking)
    # ==========================================
    if visual_level == 1:
        plan_bubble, inf_bubble = None, None
        f_n = pygame.font.SysFont("arial", 17)
        f_b = pygame.font.SysFont("arial", 18, bold=True) # 볼드 크기 보정

        if thought_msg:
            lines = thought_msg.split('\n')
            for line in lines:
                is_p1_plan = "Plan for Player 1" in line
                is_p0_inf = "Intention for Player 0" in line
                if not (is_p1_plan or is_p0_inf): continue

                content_match = re.search(r'"([^"]+)"', line)
                if not content_match: continue
                raw_c = content_match.group(1)

                # 액션 종류 파싱
                found_skill = "wait"
                skills = ["pickup_onion", "pickup_dish", "put_onion_in_pot", "fill_dish_with_soup", "deliver_soup", "pickup_tomato"]
                for s in skills:
                    if s in raw_c.lower(): found_skill = s; break
                
                idx_match = re.search(r'\((\d+)\)', raw_c)
                idx = int(idx_match.group(1)) if idx_match else 0
                
                # 자연어 변환 및 리치 텍스트 렌더링
                display_text = transform_to_english_natural(found_skill, idx, is_p0_inf, [])
                surf = render_rich_text(display_text, f_n, f_b)

                if is_p1_plan:
                    plan_bubble = {"surf": surf, "is_thought": False, "color": (20, 20, 20)}
                elif is_p0_inf and show_intention:
                    inf_bubble = {"surf": surf, "is_thought": True, "color": (130, 130, 130)}

        # 말풍선 리스트 (Plan이 아래, Intention이 위)
        bubbles = [b for b in [plan_bubble, inf_bubble] if b]
        px, py = get_player_screen_pos(num_AI, env, map_start_y, start_x, surf_width, surf_height)
        
        y_off = 75
        for i, b in enumerate(bubbles):
            b_h = draw_speech_bubble(window, b['surf'], px, py, is_thought=b['is_thought'], 
                                     border_color=b['color'], alpha=220, y_offset=y_off, draw_tail=(i==0))
            y_off += (b_h + 20)

        # 레벨 1 범례 박스
        leg_x, leg_y = screen_width - 270, screen_height - 120
        pygame.draw.rect(window, (245, 245, 245), (leg_x-5, leg_y-5, 260, 110), border_radius=8)
        pygame.draw.rect(window, (200, 200, 200), (leg_x-5, leg_y-5, 260, 110), 1, border_radius=8)
        
        window.blit(f_small.render("Black Border: AI's Next Plan", True, (0,0,0)), (leg_x + 20, leg_y + 5))
        window.blit(f_small.render("Grey Border: Inferred Human Intention", True, (0,0,0)), (leg_x + 20, leg_y + 30))
        window.blit(f_desc.render("System shows Plan and/or Intention", True, (80,80,80)), (leg_x + 5, leg_y + 60))
        window.blit(f_desc.render("based on the current context.", True, (80,80,80)), (leg_x + 5, leg_y + 75))

    # ==========================================
    # 4. [Visual Level 2] 타일 하이라이트 + 범례
    # ==========================================
    elif visual_level == 2:
        inf_coords, plan_coords = [], []
        if thought_msg:
            inf_coords, plan_coords = parse_separate_highlights(thought_msg, layout_dict, num_AI=num_AI)
        
        plan_color, inf_color = (80, 220, 150), (50, 120, 255) # 초록(Plan), 파랑(Inf)

        # Plan 하이라이트 (Green)
        for (hx, hy) in plan_coords:
            dx, dy = start_x + (hx * tile_w), map_start_y + (hy * tile_h)
            s = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
            s.fill((*plan_color, 140)); window.blit(s, (dx, dy))
            pygame.draw.rect(window, plan_color, (dx, dy, tile_w, tile_h), 4)

        # Intention 하이라이트 (Blue)
        if show_intention and inf_coords:
            for (hx, hy) in inf_coords:
                dx, dy = start_x + (hx * tile_w), map_start_y + (hy * tile_h)
                s = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                s.fill((*inf_color, 140)); window.blit(s, (dx, dy))
                pygame.draw.rect(window, inf_color, (dx, dy, tile_w, tile_h), 4)
                # 물음표 아이콘
                q_surf = pygame.font.SysFont("arial", 22, bold=True).render("?", True, inf_color)
                window.blit(q_surf, (dx + tile_w//2 - 5, dy + tile_h//2 - 12))

        # 레벨 2 범례 박스
        leg_x, leg_y = screen_width - 270, screen_height - 120
        pygame.draw.rect(window, (245, 245, 245), (leg_x-5, leg_y-5, 260, 110), border_radius=8)
        pygame.draw.rect(window, (200, 200, 200), (leg_x-5, leg_y-5, 260, 110), 1, border_radius=8)
        
        pygame.draw.rect(window, plan_color, (leg_x+5, leg_y+8, 15, 15))
        window.blit(f_small.render("Green: AI's Target Destination", True, (0,0,0)), (leg_x + 30, leg_y + 5))
        
        pygame.draw.rect(window, inf_color, (leg_x+5, leg_y+33, 15, 15))
        window.blit(f_small.render("Blue: Inferred Human Destination", True, (0,0,0)), (leg_x + 30, leg_y + 30))
        
        window.blit(f_desc.render("Highlights update dynamically or", True, (80,80,80)), (leg_x + 5, leg_y + 60))
        window.blit(f_desc.render("remain single based on AI's confidence.", True, (80,80,80)), (leg_x + 5, leg_y + 75))

    pygame.display.flip()



def draw_speech_bubble(window, content_surf, target_x, target_y, is_thought=False, 
                       border_color=(0, 0, 0), border_width=2, alpha=200, 
                       y_offset=75, padding=8, draw_tail=True):
    """
    말풍선을 그리고 사용된 총 높이를 반환합니다.
    """
    bw = content_surf.get_width() + (padding * 2)
    bh = content_surf.get_height() + (padding * 2)
    
    # 위치 계산 (중앙 정렬)
    bx = target_x - (bw / 2)
    by = target_y - bh - y_offset
    
    # 투명도 지원 Surface
    temp_surf = pygame.Surface((int(bw), int(bh + 15)), pygame.SRCALPHA)
    
    # 1. 배경 및 테두리 색상 설정
    bg_rgba = (255, 255, 255, alpha)
    line_rgba = (*border_color, alpha) if not is_thought else (120, 120, 120, alpha)

    # 2. 배경 사각형 (둥근 모서리)
    pygame.draw.rect(temp_surf, bg_rgba, (0, 0, bw, bh), border_radius=10)
    pygame.draw.rect(temp_surf, line_rgba, (0, 0, bw, bh), border_width, border_radius=10)

    # 3. 꼬리(Tail) 그리기
    if draw_tail:
        t_w, t_h = 10, 10
        t_points = [(bw//2 - t_w, bh), (bw//2 + t_w, bh), (bw//2, bh + t_h)]
        pygame.draw.polygon(temp_surf, bg_rgba, t_points)
        pygame.draw.lines(temp_surf, line_rgba, False, [t_points[0], t_points[2], t_points[1]], border_width)

    # 4. 내용물 blit (텍스트 자체에도 alpha 적용)
    content_surf.set_alpha(alpha)
    temp_surf.blit(content_surf, (padding, padding))
    
    window.blit(temp_surf, (bx, by))
    
    return bh  # 그린 말풍선의 높이를 반환하여 다음 오프셋 계산에 사용

def run_tutorial(env, screen, visualizer, ai_agent=None, args=None):
    clock = pygame.time.Clock()
    layout_dict = generate_layout_dict(env.mdp)
    served, step = 0, 0
    current_thought = ""

    # AI 초기화 및 첫 번째 생각 생성
    if ai_agent:
        ai_agent.async_mode = args.is_async
        # AI가 있다면 첫 스텝의 생각을 미리 계산
        try:
            _ = ai_agent.action(env.state) 
            current_thought = getattr(ai_agent, 'current_thought', "")
        except:
            current_thought = "AI is thinking..."
    else:
        current_thought = "Single Player Mode - Follow the guide!"

    # [핵심] 창이 열리자마자 첫 프레임을 그림
    render_game(screen, visualizer, env, step, env.horizon, served, 1, 
                args.visual_level, layout_dict, thought_msg=current_thought)

    while served < 2:
        human_action = Action.STAY
        act_chosen = False
        start_t = pygame.time.get_ticks()
        
        # 입력 대기 (delay 동안 반복)
        while pygame.time.get_ticks() - start_t < args.delay:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and not act_chosen:
                    if event.key == pygame.K_UP: human_action = Direction.NORTH
                    elif event.key == pygame.K_DOWN: human_action = Direction.SOUTH
                    elif event.key == pygame.K_LEFT: human_action = Direction.WEST
                    elif event.key == pygame.K_RIGHT: human_action = Direction.EAST
                    elif event.key == pygame.K_SPACE: human_action = Action.INTERACT
                    if human_action != Action.STAY: act_chosen = True
            
            # 비실시간 모드에서 입력 전까지 화면 계속 렌더링 (멈춤 방지)
            if not args.is_async and not act_chosen:
                start_t = pygame.time.get_ticks()
            elif not args.is_async and act_chosen:
                break
            clock.tick(60)

        # AI 행동 및 환경 업데이트
        ai_action = Action.STAY
        if ai_agent:
            ai_action = ai_agent.action(env.state, partner_action=human_action)
            current_thought = getattr(ai_agent, 'current_thought', "")

        _, reward, _, _ = env.step((human_action, ai_action))
        if reward > 0: served += 1
        step += 1

        # 화면 업데이트
        render_game(screen, visualizer, env, step, env.horizon, served, 1, 
                    args.visual_level, layout_dict, thought_msg=current_thought)

    print("Tutorial finished!")
    pygame.time.wait(2000)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi', type=boolean_argument, default=True)
    parser.add_argument('--async', dest='is_async', type=boolean_argument, default=True)
    parser.add_argument('--visual_level', type=int, default=1) # 0: None, 1: NL, 2: Highlight
    parser.add_argument('--delay', type=int, default=400)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((900, 750))
    pygame.display.set_caption("Overcooked Tutorial System")
    
    tutorial_grid = ["XXXXPXXXX", "O       D", "X      2X", "X   1   X", "XXXXSXXXX"]
    mdp = OvercookedGridworld.from_grid(tutorial_grid)
    env = OvercookedEnv(mdp=mdp, horizon=9999) 
    visualizer = StateVisualizer()

    ai_agent = None
    if args.multi:
        try:
            # ProMediumLevelAgent 또는 이를 상속받은 에이전트 생성
            ai_agent = make_agent('EIRAAsync', mdp, "cramped_room", K=1)
            ai_agent.set_agent_index(1)
            ai_agent.reset()
        except Exception as e:
            print(f"AI 로드 실패: {e}")

    run_tutorial(env, screen, visualizer, ai_agent=ai_agent, args=args)
    pygame.quit()