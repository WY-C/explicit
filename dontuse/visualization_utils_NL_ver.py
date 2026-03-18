import pygame
import re
import platform
import os

# LLM output 자연스럽게 변경 (텍스트만 처리)
def render_rich_text(text, font_normal, font_bold, color=(0, 0, 0)):
    """
    텍스트 내의 **단어** 부분을 볼드체로, 나머지는 일반 폰트로 렌더링하여 하나의 Surface로 반환합니다.
    """
    parts = text.split('**')
    rendered_parts = []
    total_width = 0
    max_height = 0

    for i, part in enumerate(parts):
        if not part: continue
        # 짝수 인덱스는 일반, 홀수 인덱스는 볼드
        is_bold = (i % 2 != 0)
        font = font_bold if is_bold else font_normal
        surf = font.render(part, True, color)
        
        rendered_parts.append(surf)
        total_width += surf.get_width()
        max_height = max(max_height, surf.get_height())

    if total_width == 0:
        return font_normal.render(text, True, color)

    # 파트들을 이어붙일 투명 배경의 최종 Surface 생성
    final_surf = pygame.Surface((total_width, max_height), pygame.SRCALPHA)
    current_x = 0
    for surf in rendered_parts:
        final_surf.blit(surf, (current_x, 0))
        current_x += surf.get_width()

    return final_surf

def transform_to_english_natural(skill_name, idx, is_thought, target_objects):
    direction_str = ""
    
    # 💡 [핵심] 두 개의 오브젝트가 있고 인덱스가 유효할 때 좌표를 비교합니다.
    if target_objects and len(target_objects) == 2 and idx is not None and 0 <= idx < 2:
        # 오브젝트가 딕셔너리(position 포함)인지 단순 튜플 좌표인지 판별하여 좌표 추출
        def get_pos(obj):
            return obj['position'] if isinstance(obj, dict) else obj
        
        target_pos = get_pos(target_objects[idx])
        other_pos = get_pos(target_objects[1 - idx])
        
        x_diff = target_pos[0] - other_pos[0]
        y_diff = target_pos[1] - other_pos[1]
        
        # 가로(X축) 차이가 세로(Y축) 차이보다 크거나 같으면 좌/우 사용
        if abs(x_diff) >= abs(y_diff):
            direction_str = "오른쪽 " if x_diff > 0 else "왼쪽 "
        # 세로(Y축) 차이가 더 크면 위/아래 사용 (Y는 아래로 갈수록 커짐)
        else:
            direction_str = "아래쪽 " if y_diff > 0 else "위쪽 "

    # AI 자신의 행동 계획
    if not is_thought: 
        skill_map = {
            "pickup_onion": f"내가 **{direction_str}양파**를 집을게.",
            "pickup_dish": f"내가 **{direction_str}접시**를 가져올게.",
            "pickup_tomato": f"내가 **{direction_str}토마토**를 가져올게.",
            "put_onion_in_pot": f"내가 이걸 **{direction_str}냄비**에 넣을게.",
            "put_tomato_in_pot": f"내가 이걸 **{direction_str}냄비**에 넣을게.",
            "fill_dish_with_soup": f"내가 **{direction_str}수프**를 그릇에 담을게.",
            "deliver_soup": "내가 **수프**를 서빙할게.",
            "place_obj_on_counter": "내가 이걸 **카운터**에 둘게.",
            "wait": "내가 잠깐 **대기**할게."
        }
    # 파트너(사람) 행동 예측
    else:
        skill_map = {
            "pickup_onion": f"네가 **{direction_str}양파**를 집으려는 것 같네.",
            "pickup_dish": f"네가 **{direction_str}접시**를 가져오려는 것 같네.",
            "pickup_tomato": f"네가 **{direction_str}토마토**를 가져오려는 것 같네.",
            "put_onion_in_pot": f"네가 그걸 **{direction_str}냄비**에 넣으려는 것 같네.",
            "put_tomato_in_pot": f"네가 그걸 **{direction_str}냄비**에 넣으려는 것 같네.",
            "fill_dish_with_soup": f"네가 **{direction_str}수프**를 담으려는 것 같네.",
            "deliver_soup": "네가 **수프**를 서빙하려는 것 같네.",
            "place_obj_on_counter": "네가 그걸 **카운터**에 두려는 것 같네.",
            "wait": "네가 잠깐 **대기**하려는 것 같네."
        }
    
    return skill_map.get(skill_name, "생각 중...")

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
        if not hasattr(mdp, f"get_{key}_locations"):
            continue
            
        locations = getattr(mdp, f"get_{key}_locations")()
        
        if not locations:
            continue
            
        layout_data[key] = []

        for i, pos in enumerate(locations):
            item_info = {
                "id": i,
                "type": readable_name,            
                "full_name": f"<{readable_name} {i}>", 
                "position": pos                   
            }
            layout_data[key].append(item_info)

    return layout_data

# highlight 좌표 파싱 (캐싱 기능 추가)
def parse_separate_highlights(thought_text, layout_dict, num_AI=None):
    """
    LLM의 사고 과정(thought_text)을 파싱하여 하이라이트할 좌표를 반환합니다.
    wait 액션일 경우 직전의 하이라이트 좌표를 유지합니다.
    """
    # 💡 함수에 변수를 저장하여 이전 좌표를 기억(캐싱)합니다.
    if not hasattr(parse_separate_highlights, "cached_inf"):
        parse_separate_highlights.cached_inf = []
    if not hasattr(parse_separate_highlights, "cached_plan"):
        parse_separate_highlights.cached_plan = []

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
        match = re.search(r'(\w+)\(?(\d+)?\)?', act_str) 
        if match:
            act_name = match.group(1)
            idx_str = match.group(2)
            if not idx_str: return [] 
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
        coords = _action_to_coords(action_str)
        
        # 💡 wait이면 캐시된 좌표를 불러오고, 아니면 새 좌표를 저장합니다.
        if "wait" in action_str.lower():
            highlight_for_inference_coords = parse_separate_highlights.cached_inf
        else:
            highlight_for_inference_coords = coords
            parse_separate_highlights.cached_inf = coords

    # --- Plan (나) 파싱 ---
    plan_match = re.search(r'Plan.*?(?:Player (\d+))?.*:\s*"([^"]+)"', thought_text, re.IGNORECASE)
    if plan_match:
        parsed_id = plan_match.group(1)
        action_str = plan_match.group(2)
        coords = _action_to_coords(action_str)
        
        # 💡 wait이면 캐시된 좌표를 불러오고, 아니면 새 좌표를 저장합니다.
        if "wait" in action_str.lower():
            highlight_for_plan_coords = parse_separate_highlights.cached_plan
        else:
            highlight_for_plan_coords = coords
            parse_separate_highlights.cached_plan = coords

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

# 함수 인자에 padding=4 추가
# 함수 인자에 draw_tail=True 추가
def draw_speech_bubble(window, content_surf, target_x, target_y, is_thought=False, border_color=(0, 0, 0), border_width=3, alpha=210, y_offset=75, padding=4, draw_tail=True):
    """
    말풍선을 그립니다.
    """
    bubble_w = content_surf.get_width() + (padding * 2)
    bubble_h = content_surf.get_height() + (padding * 2)
    
    bubble_x = target_x - (bubble_w / 2)
    bubble_y = target_y - (bubble_h / 2) - y_offset
    
    tail_direction = "up" if bubble_y < 10 else "down"
    if bubble_y < 10: bubble_y = target_y + 30

    temp_surf = pygame.Surface((int(bubble_w), int(bubble_h + 35)), pygame.SRCALPHA)
    
    # 기본 테두리 색상
    final_color = border_color
    if border_color == (0, 0, 0):
        final_color = (150, 150, 150) if is_thought else (0, 0, 0)

    # 1. 하얀색 배경 칠하기
    pygame.draw.rect(temp_surf, (255, 255, 255), (0, 0, bubble_w, bubble_h), 0, border_radius=8)
    
    # 2. 테두리 그리기
    pygame.draw.rect(temp_surf, final_color, (0, 0, bubble_w, bubble_h), border_width, border_radius=8)

    # 3. 꼬리 그리기 (draw_tail이 True일 때만)
    if draw_tail:
        mid_x, offset = bubble_w // 2, 6
        if tail_direction == "down":
            p1 = (mid_x, bubble_h + 12) 
            p2 = (mid_x - offset, bubble_h - 1)
            p3 = (mid_x + offset, bubble_h - 1)
        else:
            p1 = (mid_x, -12)
            p2 = (mid_x - offset, 5)
            p3 = (mid_x + offset, 5)
        
        pygame.draw.polygon(temp_surf, (255, 255, 255), [p1, p2, p3])
        pygame.draw.polygon(temp_surf, final_color, [p1, p2, p3], border_width)

    # 4. 내용물 얹기
    temp_surf.blit(content_surf, (padding, padding))
    temp_surf.set_alpha(alpha)
    window.blit(temp_surf, (bubble_x, bubble_y))

def render_game(window, visualizer, env, step, target_score, reward, num_AI, visual_level, layout_dict,
                thought_msg=None, show_intention=True):
    if not window or not visualizer:
        return

    # 💡 [추가] OS별 한글 지원 기본 폰트 자동 설정
    os_name = platform.system()
    if os_name == 'Windows':
        korean_font_name = 'malgungothic'  
    elif os_name == 'Darwin':
        korean_font_name = 'applegothic'   
    else:
        korean_font_name = 'nanumgothic'   

    # 1. [Pre-calculation] 맵 전체 오브젝트 정보 파악
    highlight_color_green = (80, 220, 150) 
    highlight_color_blue = (50, 120, 255)  
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
    
    font_header = pygame.font.SysFont(korean_font_name, 30, bold=True)
    info_text = font_header.render(f"Step: {step} | 점수: {reward} / {target_score}", True, (0, 0, 0))
    window.blit(info_text, (10, 10))

    # 3. 맵 렌더링
    map_start_y = 100
    state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
    surf_width, surf_height = state_surface.get_size()
    start_x = (screen_width - surf_width) // 2
    window.blit(state_surface, (start_x, map_start_y))

    grid_width, grid_height = len(env.mdp.terrain_mtx[0]), len(env.mdp.terrain_mtx)
    tile_w, tile_h = surf_width / grid_width, surf_height / grid_height

    # 💡 [추가] 유저의 기존 아이콘 매핑 (Overcooked 스프라이트 기준)
    icon_map = {
        "onion": "onion",
        "dish": "dish",
        "tomato": "tomato",
        "soup": "soup",
        "pot": "pot",
        "serve": "soup" 
    }

    # 5. [Speech Bubbles] 말풍선 렌더링 (💡 Level 1 전용: 게임 리소스 아이콘 + 자연어)
    if visual_level == 1:
        bubbles_to_draw = []
        
        if thought_msg:
            raw_lines = thought_msg.split('\n')

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
                    b_color, b_width = (0, 0, 0), 2 
                    
                    idx_match = re.search(r'\((\d+)\)', content_str)
                    idx = int(idx_match.group(1)) if idx_match else None
                    lower_content = content_str.lower()

                    target_skill, target_char = None, None
                    for skill, char in action_priorities:
                        if skill in lower_content:
                            target_skill, target_char = skill, char
                            break
                    
                    target_objects = obj_info.get(target_char, []) if target_char else []

                    # 1️⃣ 텍스트 렌더링
                    display_text = transform_to_english_natural(target_skill, idx, is_thought, target_objects)
                    font_normal = pygame.font.SysFont(korean_font_name, 18, bold=False)
                    font_bold = pygame.font.SysFont(korean_font_name, 18, bold=True)
                    text_surf = render_rich_text(display_text, font_normal, font_bold)

                    # 2️⃣ 기존 아이콘 로직으로 게임 스프라이트 불러오기
                    found_icon = next((val for key, val in icon_map.items() if key in lower_content), None)
                    icon_surf = None
                    
                    if found_icon:
                        temp = pygame.Surface((40, 40), pygame.SRCALPHA)
                        for src in [visualizer.OBJECTS_IMG, visualizer.TERRAINS_IMG]:
                            try: 
                                src.blit_on_surface(temp, (0, 0), found_icon)
                                break
                            except: 
                                continue
                        # 기존 코드에서 쓰던 크기 그대로 적용
                        icon_surf = pygame.transform.scale(temp, (50, 50))

                    # 3️⃣ 아이콘과 텍스트 이어붙이기
                    content_surf = None
                    if icon_surf and text_surf:
                        combined_w = icon_surf.get_width() + text_surf.get_width() + 8 # 여백 8px
                        combined_h = max(icon_surf.get_height(), text_surf.get_height())
                        content_surf = pygame.Surface((combined_w, combined_h), pygame.SRCALPHA)
                        
                        icon_y = (combined_h - icon_surf.get_height()) // 2
                        text_y = (combined_h - text_surf.get_height()) // 2
                        
                        content_surf.blit(icon_surf, (0, icon_y))
                        content_surf.blit(text_surf, (icon_surf.get_width() + 8, text_y))
                    else:
                        content_surf = text_surf

                    if content_surf:
                        bubbles_to_draw.append({"pid": target_pid, "surf": content_surf, "is_thought": is_thought, "color": b_color, "width": b_width})

        # 파싱된 말풍선이 하나도 없다면 "생각 중..." 출력
        if len(bubbles_to_draw) == 0:
            font_normal = pygame.font.SysFont(korean_font_name, 18, bold=False)
            font_bold = pygame.font.SysFont(korean_font_name, 18, bold=True)
            content_surf = render_rich_text("생각 중...", font_normal, font_bold, color=(0, 0, 0))
            bubbles_to_draw.append({
                "pid": num_AI,
                "surf": content_surf,
                "is_thought": False, 
                "color": (150, 150, 150),
                "width": 2
            })

        # 화면에 말풍선 그리기
        for b in bubbles_to_draw:
            px, py = get_player_screen_pos(num_AI, env, map_start_y, start_x, surf_width, surf_height)
            
            target_y_offset = 110 if b['is_thought'] else 85
            show_tail = not b['is_thought']
            
            draw_speech_bubble(
                window, b['surf'], px, py, 
                is_thought=b['is_thought'], 
                border_color=b['color'], 
                border_width=b['width'],
                alpha=120,
                y_offset=target_y_offset,
                draw_tail=show_tail
            )

    # 6. 🚨 [Visual Level 2] 목표 타일 하이라이트 전용 로직
    elif visual_level == 2 and thought_msg:
        try:
            highlight_for_inference_coords, highlight_for_plan_coords = parse_separate_highlights(thought_msg, layout_dict, num_AI=num_AI)
            if not show_intention: highlight_for_inference_coords = []
            
            if num_AI == 0: 
                inf_color, plan_color = highlight_color_green, highlight_color_blue 
            else: 
                inf_color, plan_color = highlight_color_blue, highlight_color_green  

            # 1. AI의 행동 계획 (Plan) 그리기
            if highlight_for_plan_coords:
                s_plan = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                s_plan.fill((*plan_color, 100))
                for (hx, hy) in highlight_for_plan_coords:
                    dx = start_x + (hx * tile_w)
                    dy = map_start_y + (hy * tile_h)
                    window.blit(s_plan, (dx, dy))
                    pygame.draw.rect(window, plan_color, pygame.Rect(dx, dy, tile_w, tile_h), 3)

            # 2. 파트너 행동 예측 (Inference) 그리기
            if highlight_for_inference_coords:
                s_inf = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                s_inf.fill((*inf_color, 100))
                for (hx, hy) in highlight_for_inference_coords:
                    dx = start_x + (hx * tile_w)
                    dy = map_start_y + (hy * tile_h)
                    
                    window.blit(s_inf, (dx, dy))
                    pygame.draw.rect(window, inf_color, pygame.Rect(dx, dy, tile_w, tile_h), 3)

        except Exception as e:
             print(f"Error in Level 2 rendering: {e}")

    pygame.display.flip()