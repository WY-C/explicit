import pygame
import re

#LLM output 자연스럽게 변경
def transform_to_english_natural(skill_name, idx, is_thought, has_two_objs):
    """
    말풍선 텍스트를 영문 자연어로 변환합니다.
    - idx 0: blue, idx 1: red (사물이 2개일 때만 적용)
    - Plan: "I'll..." / Intention: "Do you want to...?"
    """
    # 1. 색상 형용사 결정 (has_two_objs가 True일 때만 색상 부여)
    color_str = ""
    if has_two_objs and idx is not None:
        color_str = "blue " if idx == 0 else "red "

    # 2. 스킬별 영어 문구 생성
    if not is_thought: # [Plan] AI인 나의 행동
        skill_map = {
            "pickup_onion": f"I'll grab the {color_str}onion.",
            "pickup_dish": f"I'll get the {color_str}dish.",
            "pickup_tomato": f"I'll get the {color_str}tomato.",
            "put_onion_in_pot": f"Putting it in the {color_str}pot.",
            "put_tomato_in_pot": f"Putting it in the {color_str}pot.",
            "fill_dish_with_soup": f"Plating the {color_str}soup!",
            "deliver_soup": f"Heading to the {color_str}delivery loc.",
            "place_obj_on_counter": "Placing this on the counter.",
            "wait": "I'll wait for a sec."
        }
    else: # [Intention] 파트너(사람)의 의도 예측
        skill_map = {
            "pickup_onion": f"Do you want to get the {color_str}onion?",
            "pickup_dish": f"Do you want to get the {color_str}dish?",
            "pickup_tomato": f"Do you want to get the {color_str}tomato?",
            "put_onion_in_pot": f"Do you want to put it in the {color_str}pot?",
            "put_tomato_in_pot": f"Do you want to put it in the {color_str}pot?",
            "fill_dish_with_soup": f"Do you want to plate the {color_str}soup?",
            "deliver_soup": "Do you want to deliver the soup?",
            "place_obj_on_counter": "Placing that on the counter?",
            "wait": "Are you waiting?"
        }
    
    return skill_map.get(skill_name, "Thinking...")#layout dict 생성
def generate_layout_dict(mdp):
    """
    레이아웃 정보를 문자열 대신 구조화된 딕셔너리로 반환합니다.
    (Level 3 하이라이트 기능 등에서 좌표를 찾을 때 사용)
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
        # MDP에서 위치 정보 가져오기 (예: get_onion_dispenser_locations)
        if not hasattr(mdp, f"get_{key}_locations"):
            continue
            
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
#highlight 좌표 파싱
def parse_separate_highlights(thought_text, layout_dict, num_AI=None):
    """
    LLM의 사고 과정(thought_text)을 파싱하여 하이라이트할 좌표를 반환합니다.
    Return: (highlight_for_inference_coords, highlight_for_plan_coords)
    """
    highlight_for_inference_coords = []
    highlight_for_plan_coords = []
    
    if not thought_text or not layout_dict:
        return [], []

    def _action_to_coords(act_str):
        if not act_str: return []
        key_map = {
            "pickup_onion": "onion_dispenser",
            "pickup_dish": "dish_dispenser",
            "put_onion_in_pot": "pot",
            "fill_dish_with_soup": "pot",
            "deliver_soup": "serving",
            "place_obj_on_counter": "counter" 
        }
        # 정규식: action_name(index) 또는 action_name
        match = re.search(r'(\w+)\(?(\d+)?\)?', act_str) 
        if match:
            act_name = match.group(1)
            idx_str = match.group(2)
            
            # 인자가 없는 스킬(wait 등)은 좌표 표시 불가
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
    intention_match = re.search(r'Intention.*?(?:Player (\d+))?.*:\s*"([^"]+)"', thought_text, re.IGNORECASE)
    if intention_match:
        action_str = intention_match.group(2)
        highlight_for_inference_coords = _action_to_coords(action_str)

    # --- Plan (나) 파싱 ---
    plan_match = re.search(r'Plan.*?(?:Player (\d+))?.*:\s*"([^"]+)"', thought_text, re.IGNORECASE)
    if plan_match:
        parsed_id = plan_match.group(1) # '0' 또는 '1'
        action_str = plan_match.group(2)
        
        # 플레이어 ID 검증 (경고 출력)
        if num_AI is not None and parsed_id is not None:
            if int(parsed_id) != num_AI:
                print(f"Warning: Parsed plan for Player {parsed_id}, but I am Player {num_AI}.")
        
        highlight_for_plan_coords = _action_to_coords(action_str)

    return highlight_for_inference_coords, highlight_for_plan_coords


#플레이어 스크린 좌표
def get_player_screen_pos(player_idx, env, map_start_y, start_x, surf_width, surf_height):
    """플레이어의 현재 화면상 중심 좌표(x, y)를 반환합니다."""
    player = env.state.players[player_idx]
    grid_pos = player.position # (x, y)
    
    grid_width = len(env.mdp.terrain_mtx[0])
    grid_height = len(env.mdp.terrain_mtx)
    
    tile_w = surf_width / grid_width
    tile_h = surf_height / grid_height
    
    # 그리드 좌표를 픽셀 좌표로 변환 (타일의 정중앙)
    screen_x = start_x + (grid_pos[0] * tile_w) + (tile_w / 2)
    screen_y = map_start_y + (grid_pos[1] * tile_h) + (tile_h / 2)
    
    return screen_x, screen_y

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

def draw_speech_bubble(window, content_surf, target_x, target_y, is_thought=False, border_color=(0, 0, 0), border_width=2, alpha=180, y_offset=75):
    """
    y_offset: 값이 클수록 플레이어 머리에서 멀어집니다 (더 위로 올라감).
    """
    padding = 4
    bubble_w = content_surf.get_width() + (padding * 2)
    bubble_h = content_surf.get_height() + (padding * 2)
    
    # 🚨 y_offset 적용 (AI는 이 값을 작게 줘서 아래로 내릴 예정)
    bubble_x = target_x - (bubble_w / 2)
    bubble_y = target_y - (bubble_h / 2) - y_offset
    
    tail_direction = "up" if bubble_y < 10 else "down"
    if bubble_y < 10: bubble_y = target_y + 30

    temp_surf = pygame.Surface((int(bubble_w), int(bubble_h + 35)), pygame.SRCALPHA)
    
    if border_color == (0, 0, 0):
        final_color, final_width = ((150, 150, 150) if is_thought else (0, 0, 0)), 2
    else:
        final_color, final_width = border_color, border_width

    # 1. 말풍선 배경 및 테두리
    pygame.draw.rect(temp_surf, (255, 255, 255), (0, 0, bubble_w, bubble_h), 0, border_radius=8)
    pygame.draw.rect(temp_surf, final_color, (0, 0, bubble_w, bubble_h), final_width, border_radius=8)

    # 2. 꼬리 그리기 (y_offset에 따라 꼬리 끝점 p1을 유동적으로 조절)
    mid_x, offset = bubble_w // 2, 6
    if tail_direction == "down":
        # 꼬리 끝점 p1을 플레이어 머리 근처(y_offset보다 살짝 아래)로 고정
        p1 = (mid_x, bubble_h + (y_offset - 40)) 
        p2 = (mid_x - offset, bubble_h - 1)
        p3 = (mid_x + offset, bubble_h - 1)
    else:
        p1 = (mid_x, 0)
        p2 = (mid_x - offset, 5)
        p3 = (mid_x + offset, 5)
    
    pygame.draw.polygon(temp_surf, (255, 255, 255), [p1, p2, p3])
    pygame.draw.polygon(temp_surf, final_color, [p1, p2, p3], final_width)

    temp_surf.blit(content_surf, (padding, padding))
    temp_surf.set_alpha(alpha)
    window.blit(temp_surf, (bubble_x, bubble_y))

def render_game(window, visualizer, env, step, horizon, reward, num_AI, visual_level, layout_dict,
                thought_msg=None, show_intention=True):
    if not window or not visualizer:
        return

    # 1. [Pre-calculation] 맵 전체 오브젝트 정보 파악
    highlight_color_green = (80, 220, 150) # 보통 Plan 색상
    highlight_color_blue = (50, 120, 255)  # 보통 Inference 색상
    target_chars = ['P', 'O', 'D', 'S', 'T']
    obj_info = {char: [] for char in target_chars}
    for y, row in enumerate(env.mdp.terrain_mtx):
        for x, tile in enumerate(row):
            if tile in target_chars:
                obj_info[tile].append((x, y))
    for char in target_chars:
        obj_info[char].sort()

    # 2. 배경 및 기본 정보 렌더링
    window.fill((255, 255, 255)) 
    screen_width, screen_height = window.get_size()
    font_name = "arial"
    font_header = pygame.font.SysFont(font_name, 30, bold=True)
    info_text = font_header.render(f"Step: {step}/{horizon} | Reward: {reward}", True, (0, 0, 0))
    window.blit(info_text, (10, 10))

    # 3. 맵 렌더링
    map_start_y = 100
    state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
    surf_width, surf_height = state_surface.get_size()
    start_x = (screen_width - surf_width) // 2
    window.blit(state_surface, (start_x, map_start_y))

    grid_width, grid_height = len(env.mdp.terrain_mtx[0]), len(env.mdp.terrain_mtx)
    tile_w, tile_h = surf_width / grid_width, surf_height / grid_height

    # 4. [Map Borders] 맵 위 오브젝트 테두리는 유지 (식별용)
    if visual_level in [1, 2]:
        for char, locs in obj_info.items():
            if len(locs) == 2:
                for i, (hx, hy) in enumerate(locs):
                    color = (0, 120, 255) if i == 0 else (255, 60, 60)
                    pygame.draw.rect(window, color, pygame.Rect(start_x + hx * tile_w, map_start_y + hy * tile_h, tile_w, tile_h), 4)

    # 5. [Speech Bubbles] 말풍선 렌더링 (Level 1, 2 전용)
    if thought_msg and visual_level in [1, 2]:
        raw_lines = thought_msg.split('\n')
        bubbles_to_draw = []
        
        icon_map = {
            "pickup_onion": "onions", "pickup_dish": "dishes", "pickup_tomato": "tomatoes",
            "put_onion_in_pot": "pot", "put_tomato_in_pot": "pot",
            "fill_dish_with_soup": "soup-onion-cooked",
            "deliver_soup": "serving", "wait": "stay"
        } 

        action_priorities = [
            ("put_onion_in_pot", 'P'), ("put_tomato_in_pot", 'P'), ("fill_dish_with_soup", 'P'),
            ("pickup_onion", 'O'), ("pickup_dish", 'D'), ("pickup_tomato", 'T'), ("deliver_soup", 'S'),
            ("place_obj_on_counter", 'C'), ("wait", None)
        ]

        for line in raw_lines:
            line = line.strip()
            if not line: continue
            
            target_pid, is_thought, content_str = -1, False, ""
            if "Plan" in line:
                m = re.search(r'Player (\d+)', line)
                if m: target_pid, content_str = int(m.group(1)), line.split(':')[-1].strip().replace('"', '')
            elif "Intention" in line and show_intention:
                m = re.search(r'Player (\d+)', line)
                if m: target_pid, is_thought, content_str = int(m.group(1)), True, line.split(':')[-1].strip().replace('"', '')
            
            if target_pid != -1:
                # 🚨 자연어 말풍선은 테두리(blue, red) 없이 검정/회색으로 고정 (이전 규칙 반영 완료)
                b_color, b_width = (0, 0, 0), 2 
                display_text = content_str 
                
                idx_match = re.search(r'\((\d+)\)', content_str)
                idx = int(idx_match.group(1)) if idx_match else None
                lower_content = content_str.lower()

                target_skill, target_char = None, None
                for skill, char in action_priorities:
                    if skill in lower_content:
                        target_skill, target_char = skill, char
                        break
                
                has_two = (len(obj_info.get(target_char, [])) == 2) if target_char else False

                if visual_level == 2:
                    display_text = transform_to_english_natural(target_skill, idx, is_thought, has_two)

                content_surf = None
                if visual_level == 1:
                    found_icon = next((val for key, val in icon_map.items() if key in content_str), None)
                    temp = pygame.Surface((40, 40), pygame.SRCALPHA)
                    if found_icon:
                        for src in [visualizer.OBJECTS_IMG, visualizer.TERRAINS_IMG]:
                            try: src.blit_on_surface(temp, (0, 0), found_icon); break
                            except: continue
                    content_surf = pygame.transform.scale(temp, (50, 50))
                else:
                    font_bubble = pygame.font.SysFont("arial", 18, bold=True)
                    content_surf = font_bubble.render(display_text, True, (0, 0, 0))

                if content_surf:
                    bubbles_to_draw.append({"pid": target_pid, "surf": content_surf, "is_thought": is_thought, "color": b_color, "width": b_width})

        for b in bubbles_to_draw:
            px, py = get_player_screen_pos(b['pid'], env, map_start_y, start_x, surf_width, surf_height)
            target_y_offset = 60 if b['pid'] == num_AI else 95
            
            draw_speech_bubble(
                window, b['surf'], px, py, 
                is_thought=b['is_thought'], 
                border_color=b['color'], 
                border_width=b['width'],
                alpha=180,
                y_offset=target_y_offset
            )

    # 6. 🚨 [Visual Level 3] 목표 타일 하이라이트 전용 로직 (말풍선 없음) 🚨
    elif visual_level == 3 and thought_msg:
        # ... (기존 Level 3 코드와 동일) ...
        try:
            highlight_for_inference_coords, highlight_for_plan_coords = parse_separate_highlights(thought_msg, layout_dict, num_AI=num_AI)
            if not show_intention: highlight_for_inference_coords = []
            if num_AI == 0: inf_color, plan_color = highlight_color_green, highlight_color_blue 
            else: inf_color, plan_color = highlight_color_blue, highlight_color_green  

            for coords, color in [(highlight_for_inference_coords, inf_color), (highlight_for_plan_coords, plan_color)]:
                if not coords: continue
                s = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                s.fill((*color, 100))
                for (hx, hy) in coords:
                    dx = start_x + (hx * tile_w)
                    dy = map_start_y + (hy * tile_h)
                    window.blit(s, (dx, dy))
                    pygame.draw.rect(window, color, pygame.Rect(dx, dy, tile_w, tile_h), 3)
        except Exception as e:
             print(f"Error in Level 3 rendering: {e}")

    pygame.display.flip()


