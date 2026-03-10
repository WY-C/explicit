import sys
import os
import pygame
import re
import argparse
import platform
import json
import datetime
import time
from distutils.util import strtobool

# 오버쿡드 환경 및 에이전트 임포트
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# 커스텀 모듈 연결
try:
    from utils import make_agent
except ImportError:
    pass

# =========================================================
# 💡 [추가] 튜토리얼 데이터 저장을 위한 로거 클래스
# =========================================================
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

# =========================================================
# OS별 한글 폰트 설정
# =========================================================
def get_korean_font_name():
    os_name = platform.system()
    if os_name == 'Windows': return 'malgungothic'
    elif os_name == 'Darwin': return 'applegothic'
    else: return 'nanumgothic'

K_FONT = get_korean_font_name()

# =========================================================
# Utility Functions (Rendering & Parsing)
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
        y_pos = (max_height - surf.get_height()) // 2
        final_surf.blit(surf, (current_x, y_pos))
        current_x += surf.get_width()

    return final_surf

def transform_to_korean_natural(skill_name, idx, is_thought, target_objects):
    direction_str = ""
    
    if target_objects and len(target_objects) == 2 and idx is not None and 0 <= idx < 2:
        def get_pos(obj): return obj['position'] if isinstance(obj, dict) else obj
        target_pos = get_pos(target_objects[idx])
        other_pos = get_pos(target_objects[1 - idx])
        
        x_diff = target_pos[0] - other_pos[0]
        y_diff = target_pos[1] - other_pos[1]
        
        if abs(x_diff) >= abs(y_diff):
            direction_str = "오른쪽 " if x_diff > 0 else "왼쪽 "
        else:
            direction_str = "아래쪽 " if y_diff > 0 else "위쪽 "

    if not is_thought: 
        skill_map = {
            "pickup_onion": f"내가 **{direction_str}양파**를 집을게...",
            "pickup_dish": f"내가 **{direction_str}접시**를 가져올게...",
            "pickup_tomato": f"내가 **{direction_str}토마토**를 가져올게...",
            "put_onion_in_pot": f"내가 이걸 **{direction_str}냄비**에 넣을게...",
            "put_tomato_in_pot": f"내가 이걸 **{direction_str}냄비**에 넣을게...",
            "fill_dish_with_soup": f"내가 **{direction_str}수프**를 그릇에 담을게...",
            "deliver_soup": "내가 **수프**를 서빙할게...",
            "place_obj_on_counter": "내가 이걸 **카운터**에 둘게...",
            "wait": "내가 잠깐 **대기**할게..."
        }
    else:
        skill_map = {
            "pickup_onion": f"네가 **{direction_str}양파**를 집으려는 것 같네...",
            "pickup_dish": f"네가 **{direction_str}접시**를 가져오려는 것 같네...",
            "pickup_tomato": f"네가 **{direction_str}토마토**를 가져오려는 것 같네...",
            "put_onion_in_pot": f"네가 그걸 **{direction_str}냄비**에 넣으려는 것 같네...",
            "put_tomato_in_pot": f"네가 그걸 **{direction_str}냄비**에 넣으려는 것 같네...",
            "fill_dish_with_soup": f"네가 **{direction_str}수프**를 담으려는 것 같네...",
            "deliver_soup": "네가 **수프**를 서빙하려는 것 같네...",
            "place_obj_on_counter": "네가 그걸 **카운터**에 두려는 것 같네...",
            "wait": "네가 잠깐 **대기**하려는 것 같네..."
        }
    return skill_map.get(skill_name, "생각 중...")

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
        idx_match = re.search(r'\((\d+)\)', text)
        idx = int(idx_match.group(1)) if idx_match else 0
        
        target_key = None
        lower_text = text.lower()
        
        if "pickup_onion" in lower_text: target_key = "onion_dispenser"
        elif "pickup_dish" in lower_text: target_key = "dish_dispenser"
        elif "put_onion" in lower_text or "put_tomato" in lower_text: target_key = "pot"
        elif "fill_dish_with_soup" in lower_text: target_key = "pot"
        elif "deliver_soup" in lower_text: target_key = "serving"

        if target_key and target_key in layout_dict and idx < len(layout_dict[target_key]):
            return [layout_dict[target_key][idx]['position']]
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


def render_game(window, visualizer, env, step, horizon, reward, num_AI, visual_level, layout_dict,
                thought_msg=None, show_intention=True):
    if not window or not visualizer: return

    window.fill((255, 255, 255)) 
    screen_width, screen_height = window.get_size()
    
    f_header = pygame.font.SysFont(K_FONT, 26, bold=True)
    header_surf = f_header.render(f"스텝: {step} | 목표: 수프 2그릇 서빙 (현재 {reward}그릇)", True, (40, 40, 40))
    window.blit(header_surf, (25, 20))

    map_start_y = 100
    state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
    surf_width, surf_height = state_surface.get_size()
    start_x = (screen_width - surf_width) // 2
    window.blit(state_surface, (start_x, map_start_y))

    grid_width = len(env.mdp.terrain_mtx[0])
    grid_height = len(env.mdp.terrain_mtx)
    tile_w = surf_width / grid_width
    tile_h = surf_height / grid_height
    
    f_title = pygame.font.SysFont(K_FONT, 20, bold=True)
    f_small = pygame.font.SysFont(K_FONT, 14)
    f_step = pygame.font.SysFont(K_FONT, 15, bold=True)
    f_desc = pygame.font.SysFont(K_FONT, 12, italic=True)

    guide_x, guide_y = 20, 470
    pygame.draw.rect(window, (252, 252, 252), (guide_x, guide_y, 280, 260), border_radius=12)
    pygame.draw.rect(window, (60, 60, 60), (guide_x, guide_y, 280, 260), 2, border_radius=12)

    window.blit(f_title.render("🎮 게임 조작 안내", True, (0, 0, 0)), (guide_x + 15, guide_y + 15))
    
    pygame.draw.line(window, (200, 200, 200), (guide_x + 15, guide_y + 45), (guide_x + 265, guide_y + 45), 1)
    window.blit(f_step.render("▶ 이동: 방향키 (상하좌우)", True, (50, 50, 50)), (guide_x + 15, guide_y + 55))
    window.blit(f_step.render("▶ 행동: 스페이스바 (집기/놓기)", True, (50, 50, 50)), (guide_x + 15, guide_y + 80))

    window.blit(f_title.render("📋 튜토리얼 미션", True, (0, 0, 0)), (guide_x + 15, guide_y + 120))
    mission_steps = [
        "1. 양파 3개를 냄비에 넣습니다.",
        "2. 수프가 다 끓을 때까지 기다립니다.",
        "3. 접시를 들고 냄비에서 수프를 담습니다.",
        "4. 완성된 수프를 회색 카운터에 서빙합니다."
    ]
    for i, txt in enumerate(mission_steps):
        window.blit(f_small.render(txt, True, (80, 80, 80)), (guide_x + 15, guide_y + 150 + (i * 25)))

    target_chars = ['P', 'O', 'D', 'S', 'T']
    obj_info = {char: [] for char in target_chars}
    for y, row in enumerate(env.mdp.terrain_mtx):
        for x, tile in enumerate(row):
            if tile in target_chars:
                obj_info[tile].append((x, y))
    for char in target_chars:
        obj_info[char].sort()

    if visual_level == 1:
        plan_bubble, inf_bubble = None, None
        f_n = pygame.font.SysFont(K_FONT, 16)
        f_b = pygame.font.SysFont(K_FONT, 16, bold=True)

        if thought_msg:
            lines = thought_msg.split('\n')
            for line in lines:
                is_p1_plan = "Plan for Player 1" in line
                is_p0_inf = "Intention for Player 0" in line
                if not (is_p1_plan or is_p0_inf): continue

                content_match = re.search(r'"([^"]+)"', line)
                if not content_match: continue
                raw_c = content_match.group(1)

                found_skill = "wait"
                skills = ["pickup_onion", "pickup_dish", "put_onion_in_pot", "fill_dish_with_soup", "deliver_soup", "pickup_tomato"]
                target_char = None
                
                for s in skills:
                    if s in raw_c.lower(): 
                        found_skill = s
                        if s == "pickup_onion": target_char = 'O'
                        elif s == "pickup_dish": target_char = 'D'
                        elif s == "put_onion_in_pot": target_char = 'P'
                        elif s == "fill_dish_with_soup": target_char = 'P'
                        elif s == "deliver_soup": target_char = 'S'
                        break
                
                idx_match = re.search(r'\((\d+)\)', raw_c)
                idx = int(idx_match.group(1)) if idx_match else 0
                target_objects = obj_info.get(target_char, []) if target_char else []
                
                display_text = transform_to_korean_natural(found_skill, idx, is_p0_inf, target_objects)
                surf = render_rich_text(display_text, f_n, f_b)

                if is_p1_plan:
                    plan_bubble = {"surf": surf, "is_thought": False, "color": (20, 20, 20)}
                elif is_p0_inf and show_intention:
                    inf_bubble = {"surf": surf, "is_thought": True, "color": (130, 130, 130)}

        bubbles = [b for b in [plan_bubble, inf_bubble] if b]
        px, py = get_player_screen_pos(num_AI, env, map_start_y, start_x, surf_width, surf_height)
        
        y_off = 75
        for i, b in enumerate(bubbles):
            b_h = draw_speech_bubble(window, b['surf'], px, py, is_thought=b['is_thought'], 
                                     border_color=b['color'], alpha=220, y_offset=y_off, draw_tail=(i==0))
            y_off += (b_h + 20)

        leg_x, leg_y = screen_width - 320, screen_height - 120
        pygame.draw.rect(window, (245, 245, 245), (leg_x-5, leg_y-5, 310, 100), border_radius=8)
        pygame.draw.rect(window, (200, 200, 200), (leg_x-5, leg_y-5, 310, 100), 1, border_radius=8)
        
        window.blit(f_small.render("⬛ 검은색 테두리: AI의 다음 행동 계획", True, (0,0,0)), (leg_x + 10, leg_y + 10))
        window.blit(f_small.render("⬜ 회색 테두리: AI가 예측한 내(사람) 의도", True, (100,100,100)), (leg_x + 10, leg_y + 40))
        window.blit(f_desc.render("상황에 따라 AI의 메시지가 화면에 출력됩니다.", True, (80,80,80)), (leg_x + 10, leg_y + 70))

    elif visual_level == 2:
        inf_coords, plan_coords = [], []
        if thought_msg:
            inf_coords, plan_coords = parse_separate_highlights(thought_msg, layout_dict, num_AI=num_AI)
        
        plan_color, inf_color = (80, 220, 150), (50, 120, 255)

        for (hx, hy) in plan_coords:
            dx, dy = start_x + (hx * tile_w), map_start_y + (hy * tile_h)
            s = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
            s.fill((*plan_color, 140)); window.blit(s, (dx, dy))
            pygame.draw.rect(window, plan_color, (dx, dy, tile_w, tile_h), 4)

        if show_intention and inf_coords:
            for (hx, hy) in inf_coords:
                dx, dy = start_x + (hx * tile_w), map_start_y + (hy * tile_h)
                s = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                s.fill((*inf_color, 140)); window.blit(s, (dx, dy))
                pygame.draw.rect(window, inf_color, (dx, dy, tile_w, tile_h), 4)
                
                q_surf = pygame.font.SysFont("arial", 22, bold=True).render("?", True, inf_color)
                window.blit(q_surf, (dx + tile_w//2 - 5, dy + tile_h//2 - 12))

        leg_x, leg_y = screen_width - 320, screen_height - 120
        pygame.draw.rect(window, (245, 245, 245), (leg_x-5, leg_y-5, 310, 100), border_radius=8)
        pygame.draw.rect(window, (200, 200, 200), (leg_x-5, leg_y-5, 310, 100), 1, border_radius=8)
        
        pygame.draw.rect(window, plan_color, (leg_x+10, leg_y+12, 15, 15))
        window.blit(f_small.render("초록색 칸: AI의 이동 목표 (계획)", True, (0,0,0)), (leg_x + 35, leg_y + 10))
        
        pygame.draw.rect(window, inf_color, (leg_x+10, leg_y+42, 15, 15))
        window.blit(f_small.render("파란색 칸: AI가 예측한 내 이동 목표", True, (0,0,0)), (leg_x + 35, leg_y + 40))
        
        window.blit(f_desc.render("파란색 칸에는 ? 기호가 함께 표시됩니다.", True, (80,80,80)), (leg_x + 10, leg_y + 70))

    pygame.display.flip()


def draw_speech_bubble(window, content_surf, target_x, target_y, is_thought=False, 
                       border_color=(0, 0, 0), border_width=2, alpha=200, 
                       y_offset=75, padding=8, draw_tail=True):
    bw = content_surf.get_width() + (padding * 2)
    bh = content_surf.get_height() + (padding * 2)
    bx = target_x - (bw / 2)
    by = target_y - bh - y_offset
    
    temp_surf = pygame.Surface((int(bw), int(bh + 15)), pygame.SRCALPHA)
    
    bg_rgba = (255, 255, 255, alpha)
    line_rgba = (*border_color, alpha) if not is_thought else (120, 120, 120, alpha)

    pygame.draw.rect(temp_surf, bg_rgba, (0, 0, bw, bh), border_radius=10)
    pygame.draw.rect(temp_surf, line_rgba, (0, 0, bw, bh), border_width, border_radius=10)

    if draw_tail:
        t_w, t_h = 10, 10
        t_points = [(bw//2 - t_w, bh), (bw//2 + t_w, bh), (bw//2, bh + t_h)]
        pygame.draw.polygon(temp_surf, bg_rgba, t_points)
        pygame.draw.lines(temp_surf, line_rgba, False, [t_points[0], t_points[2], t_points[1]], border_width)

    content_surf.set_alpha(alpha)
    temp_surf.blit(content_surf, (padding, padding))
    
    window.blit(temp_surf, (bx, by))
    return bh

def run_tutorial(env, screen, visualizer, ai_agent=None, args=None):
    logger = OvercookedLogger(log_dir=args.log_dir, variant=vars(args))
    layout_dict = generate_layout_dict(env.mdp)
    served, step, inter_col_total = 0, 0, 0
    current_thought = ""

    while served < 2:
        human_action = Action.STAY
        act_chosen = False
        start_t = pygame.time.get_ticks()
        
        while pygame.time.get_ticks() - start_t < args.delay:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and not act_chosen:
                    if event.key == pygame.K_UP: human_action = Direction.NORTH
                    elif event.key == pygame.K_DOWN: human_action = Direction.SOUTH
                    elif event.key == pygame.K_LEFT: human_action = Direction.WEST
                    elif event.key == pygame.K_RIGHT: human_action = Direction.EAST
                    elif event.key == pygame.K_SPACE: human_action = Action.INTERACT
                    if human_action != Action.STAY: act_chosen = True
            if not args.is_async and act_chosen: break
            pygame.time.delay(1)

        ai_action = ai_agent.action(env.state, partner_action=human_action) if ai_agent else Action.STAY
        current_thought = getattr(ai_agent, 'current_thought', "") if ai_agent else ""

        # 💡 상호 충돌 감지
        old_p0, old_p1 = env.state.players[0].position, env.state.players[1].position
        int_p0 = (old_p0[0] + human_action[0], old_p0[1] + human_action[1]) if human_action in Direction.ALL_DIRECTIONS else old_p0
        int_p1 = (old_p1[0] + ai_action[0], old_p1[1] + ai_action[1]) if ai_action in Direction.ALL_DIRECTIONS else old_p1
        
        is_col = False
        if int_p0 == int_p1 and (human_action != Action.STAY or ai_action != Action.STAY): is_col = True
        elif int_p0 == old_p1 and int_p1 == old_p0 and human_action != Action.STAY and ai_action != Action.STAY: is_col = True
        elif human_action in Direction.ALL_DIRECTIONS and int_p0 == old_p1 and ai_action not in Direction.ALL_DIRECTIONS: is_col = True
        elif ai_action in Direction.ALL_DIRECTIONS and int_p1 == old_p0 and human_action not in Direction.ALL_DIRECTIONS: is_col = True
        if is_col: inter_col_total += 1

        _, reward, _, _ = env.step((human_action, ai_action))
        if reward > 0: served += 1
        step += 1
        
        logger.log_step({"timestep": step, "inter_collision": is_col, "reward": reward, "cumulative_reward": served})
        render_game(screen, visualizer, env, step, env.horizon, served, 1, args.visual_level, layout_dict, thought_msg=current_thought)

    return served, step, inter_col_total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str, default='cramped_room')
    parser.add_argument('--multi', type=boolean_argument, default=True)
    parser.add_argument('--async', dest='is_async', type=boolean_argument, default=True)
    parser.add_argument('--visual_level', type=int, default=1)
    parser.add_argument('--delay', type=int, default=400)
    parser.add_argument('--log_dir', type=str, default='experiments/logs')
    parser.add_argument('--name', type=str, default='tutorial')
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((900, 750))
    pygame.display.set_caption("오버쿡드 AI 협업 튜토리얼")
    
    try:
        mdp = OvercookedGridworld.from_layout_name(args.layout)
    except:
        tutorial_grid = ["XXXXPXXXX", "O       D", "X      2X", "X   1   X", "XXXXSXXXX"]
        mdp = OvercookedGridworld.from_grid(tutorial_grid)
        
    env = OvercookedEnv(mdp=mdp, horizon=9999) 
    visualizer = StateVisualizer()

    ai_agent = None
    if args.multi:
        try:
            ai_agent = make_agent('EIRAAsync', mdp, args.layout, K=1)
            ai_agent.set_agent_index(1)
            ai_agent.reset()
        except Exception as e:
            print(f"AI 로드 실패: {e}")

    run_tutorial(env, screen, visualizer, ai_agent=ai_agent, args=args)
    pygame.quit()