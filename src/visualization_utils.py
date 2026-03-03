import pygame
import re

#LLM output 자연스럽게 변경
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

def transform_to_english_natural(skill_name, idx, is_thought, has_two_objs):
    color_str = ""
    if has_two_objs and idx is not None:
        color_str = "left " if idx == 0 else "right "

    # 💡 강조하고 싶은 부분을 ** 로 감싸줍니다.
    if not is_thought: 
        skill_map = {
            "pickup_onion": f"I'll grab the **{color_str}onion**.",
            "pickup_dish": f"I'll get the **{color_str}dish**.",
            "pickup_tomato": f"I'll get the **{color_str}tomato**.",
            "put_onion_in_pot": f"I'll put this in the **{color_str}pot**.",
            "put_tomato_in_pot": f"I'll put this in the **{color_str}pot**.",
            "fill_dish_with_soup": f"I'll plate the **{color_str}soup**.",
            "deliver_soup": "I'll deliver the **soup**.",
            "place_obj_on_counter": "I'll leave this on the **counter**.",
            "wait": "I'll **wait** for a sec."
        }
    else: # [Intention] 파트너(사람)의 의도 예측 (관찰하며 혼잣말)
        skill_map = {
            "pickup_onion": f"Seems like he's going for the **{color_str}onion**...?",
            "pickup_dish": f"Seems like he's getting the **{color_str}dish**...?",
            "pickup_tomato": f"Seems like he's getting the **{color_str}tomato**...?",
            "put_onion_in_pot": f"Seems like he's putting it in the **{color_str}pot**...?",
            "put_tomato_in_pot": f"Seems like he's putting it in the **{color_str}pot**...?",
            "fill_dish_with_soup": f"Seems like he's plating the **{color_str}soup**...?",
            "deliver_soup": "Seems like he's delivering the **soup**...?",
            "place_obj_on_counter": "Seems like he's leaving it on the **counter**...?",
            "wait": "Seems like he's **waiting**...?"
        }
    
    return skill_map.get(skill_name, "Thinking...")

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
            # 꼬리 길이를 고정값(12)으로 주어 예쁘게 만듭니다.
            p1 = (mid_x, bubble_h + 12) 
            p2 = (mid_x - offset, bubble_h - 1)
            p3 = (mid_x + offset, bubble_h - 1)
        else:
            p1 = (mid_x, -12)
            p2 = (mid_x - offset, 5)
            p3 = (mid_x + offset, 5)
        
        pygame.draw.polygon(temp_surf, (255, 255, 255), [p1, p2, p3])
        pygame.draw.polygon(temp_surf, final_color, [p1, p2, p3], border_width)

    # 4. 내용물(이모지/텍스트) 얹기
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
    # if visual_level in [1, 2]:
    #     for char, locs in obj_info.items():
    #         if len(locs) == 2:
    #             for i, (hx, hy) in enumerate(locs):
    #                 color = (0, 120, 255) if i == 0 else (255, 60, 60)
    #                 pygame.draw.rect(window, color, pygame.Rect(start_x + hx * tile_w, map_start_y + hy * tile_h, tile_w, tile_h), 4)

    # 5. [Speech Bubbles] 말풍선 렌더링 (Level 1, 2 전용)
    # 5. [Speech Bubbles] 말풍선 렌더링 (Level 1, 2 전용)
    if visual_level in [1, 2]:
        bubbles_to_draw = []
        
        # thought_msg가 전달되었을 때 기존 파싱 수행
        if thought_msg:
            raw_lines = thought_msg.split('\n')
            
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
                    b_color, b_width = (0, 0, 0), 2 
                    
                    # 💡 [핵심] 기본 텍스트를 먼저 할당해 두어 에러를 방지합니다.
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

                    # 자연어 모드(Level 1)일 경우 텍스트를 자연어로 덮어씌웁니다.
                    if visual_level == 1:
                        display_text = transform_to_english_natural(target_skill, idx, is_thought, has_two)

                    content_surf = None
                    if visual_level == 2:
                        found_icon = next((val for key, val in icon_map.items() if key in content_str), None)
                        temp = pygame.Surface((40, 40), pygame.SRCALPHA)
                        if found_icon:
                            for src in [visualizer.OBJECTS_IMG, visualizer.TERRAINS_IMG]:
                                try: 
                                    src.blit_on_surface(temp, (0, 0), found_icon)
                                    break
                                except: 
                                    continue
                        
                        bounding_rect = temp.get_bounding_rect()
                        if bounding_rect.width > 0 and bounding_rect.height > 0:
                            cropped_icon = temp.subsurface(bounding_rect).copy()
                        else:
                            cropped_icon = temp
                            
                        content_surf = pygame.transform.scale(cropped_icon, (45, 45))
                    else:
                        # 텍스트 렌더링 모드일 때 커스텀 리치 텍스트 사용
                        font_normal = pygame.font.SysFont("arial", 18, bold=False)
                        font_bold = pygame.font.SysFont("arial", 18, bold=True)
                        content_surf = render_rich_text(display_text, font_normal, font_bold)

                    if content_surf:
                        bubbles_to_draw.append({"pid": target_pid, "surf": content_surf, "is_thought": is_thought, "color": b_color, "width": b_width})

        # 💡 [핵심 추가 부분] 파싱된 말풍선이 하나도 없다면 "I'm thinking..." 출력
        if len(bubbles_to_draw) == 0:
            font_bubble = pygame.font.SysFont("arial", 18, bold=True)
            # 생각 중일 때는 텍스트 색상을 약간 연한 회색(100, 100, 100)으로 해도 좋습니다.
            content_surf = font_bubble.render("I'm thinking...", True, (0, 0, 0))
            bubbles_to_draw.append({
                "pid": num_AI,
                "surf": content_surf,
                "is_thought": False,  # 꼬리를 그려주기 위해 False
                "color": (150, 150, 150), # 테두리를 회색으로 하여 구별
                "width": 2
            })

        # 화면에 말풍선 그리기 (이전 단계에서 수정한 AI 머리 위에만 띄우기 & 꼬리 제거 로직 포함)
        for b in bubbles_to_draw:
            px, py = get_player_screen_pos(num_AI, env, map_start_y, start_x, surf_width, surf_height)
            
            target_y_offset = 110 if b['is_thought'] else 85
            show_tail = not b['is_thought']
            
            draw_speech_bubble(
                window, b['surf'], px, py, 
                is_thought=b['is_thought'], 
                border_color=b['color'], 
                border_width=b['width'],
                alpha=150,
                y_offset=target_y_offset,
                draw_tail=show_tail
            )

    # 6. 🚨 [Visual Level 3] 목표 타일 하이라이트 전용 로직 (말풍선 없음) 🚨
    elif visual_level == 3 and thought_msg:
        try:
            highlight_for_inference_coords, highlight_for_plan_coords = parse_separate_highlights(thought_msg, layout_dict, num_AI=num_AI)
            if not show_intention: highlight_for_inference_coords = []
            
            # AI와 파트너 색상 지정
            if num_AI == 0: 
                inf_color, plan_color = highlight_color_green, highlight_color_blue 
            else: 
                inf_color, plan_color = highlight_color_blue, highlight_color_green  

            # 물음표 폰트 설정 (타일 크기에 맞게 큼직하게)
            font_question = pygame.font.SysFont("arial", 36, bold=True)

            # 1. AI의 행동 계획 (Plan) 그리기 - 물음표 없음
            if highlight_for_plan_coords:
                s_plan = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                s_plan.fill((*plan_color, 100))
                for (hx, hy) in highlight_for_plan_coords:
                    dx = start_x + (hx * tile_w)
                    dy = map_start_y + (hy * tile_h)
                    window.blit(s_plan, (dx, dy))
                    pygame.draw.rect(window, plan_color, pygame.Rect(dx, dy, tile_w, tile_h), 3)

            # 2. 파트너 행동 예측 (Inference) 그리기 - 타일 중앙에 물음표(?) 추가
            if highlight_for_inference_coords:
                s_inf = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                s_inf.fill((*inf_color, 100))
                for (hx, hy) in highlight_for_inference_coords:
                    dx = start_x + (hx * tile_w)
                    dy = map_start_y + (hy * tile_h)
                    
                    # 배경과 테두리 그리기
                    window.blit(s_inf, (dx, dy))
                    pygame.draw.rect(window, inf_color, pygame.Rect(dx, dy, tile_w, tile_h), 3)
                    
                    # 💡 [핵심] AI의 목표 지점(Plan)과 겹치지 않을 때만 물음표를 그립니다.
                    # if (hx, hy) not in highlight_for_plan_coords:
                    #     q_surf = font_question.render("?", True, (255, 255, 255)) 
                    #     q_shadow = font_question.render("?", True, (0, 0, 0))
                    #     shadow_rect = q_shadow.get_rect(center=(dx + tile_w//2 + 2, dy + tile_h//2 + 2))
                    #     q_rect = q_surf.get_rect(center=(dx + tile_w//2 - 3, dy + tile_h//2 - 3))

                    #     window.blit(q_shadow, shadow_rect)
                    #     window.blit(q_surf, q_rect)

        except Exception as e:
             print(f"Error in Level 3 rendering: {e}")

    pygame.display.flip()


