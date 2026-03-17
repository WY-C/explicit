import pygame
import re
import platform
import os

# LLM output 자연스럽게 변경 (Preserved for logic but not used for Level 1 icons)
def transform_to_english_natural(skill_name, idx, is_thought, has_two_objs):
    """
    Transforms the speech bubble text into natural English.
    """
    color_str = ""
    if has_two_objs and idx is not None:
        color_str = "blue " if idx == 0 else "red "

    if not is_thought: # [Plan] AI's action
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
    else: # [Intention] Predicting Partner's action
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
    
    return skill_map.get(skill_name, "Thinking...")

# Layout dict generation
def generate_layout_dict(mdp):
    layout_data = {}
    name_map = {
        "onion_dispenser": "Onion Dispenser",
        "dish_dispenser": "Dish Dispenser",
        "tomato_dispenser": "Tomato Dispenser",
        "serving": "Serving Loc",
        "pot": "Pot"
    }

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

# Highlight coordinates parsing
def parse_separate_highlights(thought_text, layout_dict, num_AI=None):
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

    intention_match = re.search(r'Intention.*?(?:Player (\d+))?.*:\s*"([^"]+)"', thought_text, re.IGNORECASE)
    if intention_match:
        action_str = intention_match.group(2)
        highlight_for_inference_coords = _action_to_coords(action_str)

    plan_match = re.search(r'Plan.*?(?:Player (\d+))?.*:\s*"([^"]+)"', thought_text, re.IGNORECASE)
    if plan_match:
        parsed_id = plan_match.group(1)
        action_str = plan_match.group(2)
        if num_AI is not None and parsed_id is not None:
            if int(parsed_id) != num_AI:
                print(f"Warning: Parsed plan for Player {parsed_id}, but I am Player {num_AI}.")
        highlight_for_plan_coords = _action_to_coords(action_str)

    return highlight_for_inference_coords, highlight_for_plan_coords

def get_player_screen_pos(player_idx, env, map_start_y, start_x, surf_width, surf_height):
    player = env.state.players[player_idx]
    grid_pos = player.position
    grid_width = len(env.mdp.terrain_mtx[0])
    grid_height = len(env.mdp.terrain_mtx)
    tile_w = surf_width / grid_width
    tile_h = surf_height / grid_height
    screen_x = start_x + (grid_pos[0] * tile_w) + (tile_w / 2)
    screen_y = map_start_y + (grid_pos[1] * tile_h) + (tile_h / 2)
    return screen_x, screen_y

def draw_centered_text(window, text, sub_text=None, color=(0, 0, 0), bg_color=(255, 255, 255)):
    if window is None: return
    window.fill(bg_color)
    screen_width, screen_height = window.get_size()
    font_name = "arial"
    main_font = pygame.font.SysFont(font_name, 30, bold=True)
    sub_font = pygame.font.SysFont(font_name, 30)
    text_surf = main_font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(screen_width // 2, screen_height // 2 - 20))
    window.blit(text_surf, text_rect)
    if sub_text:
        sub_surf = sub_font.render(sub_text, True, (100, 100, 100))
        sub_rect = sub_surf.get_rect(center=(screen_width // 2, screen_height // 2 + 40))
        window.blit(sub_surf, sub_rect)
    pygame.display.flip()

def draw_speech_bubble(window, content_surf, target_x, target_y, is_thought=False, border_color=(0, 0, 0), border_width=2, alpha=180, y_offset=75, padding=4):
    bubble_w = content_surf.get_width() + (padding * 2)
    bubble_h = content_surf.get_height() + (padding * 2)
    bubble_x = target_x - (bubble_w / 2)
    bubble_y = target_y - (bubble_h / 2) - y_offset
    tail_direction = "up" if bubble_y < 10 else "down"
    if bubble_y < 10: bubble_y = target_y + 30
    temp_surf = pygame.Surface((int(bubble_w), int(bubble_h + 35)), pygame.SRCALPHA)
    if border_color == (0, 0, 0):
        final_color, final_width = ((150, 150, 150) if is_thought else (0, 0, 0)), 2
    else:
        final_color, final_width = border_color, border_width
    pygame.draw.rect(temp_surf, (255, 255, 255), (0, 0, bubble_w, bubble_h), 0, border_radius=8)
    pygame.draw.rect(temp_surf, final_color, (0, 0, bubble_w, bubble_h), final_width, border_radius=8)
    mid_x, offset = bubble_w // 2, 6
    if tail_direction == "down":
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

def render_game(window, visualizer, env, step, target_score, reward, num_AI, visual_level, layout_dict,
                thought_msg=None, show_intention=True):
    if not window or not visualizer:
        return

    # 1. [Pre-calculation]
    target_chars = ['P', 'O', 'D', 'S', 'T']
    obj_info = {char: [] for char in target_chars}
    for y, row in enumerate(env.mdp.terrain_mtx):
        for x, tile in enumerate(row):
            if tile in target_chars:
                obj_info[tile].append((x, y))
    for char in target_chars:
        obj_info[char].sort()

    # 2. Background and Info Rendering
    window.fill((255, 255, 255)) 
    screen_width, screen_height = window.get_size()
    font_name = "arial"
    font_header = pygame.font.SysFont(font_name, 30, bold=True)
    info_text = font_header.render(f"Step: {step} | Reward: {reward} / {target_score}", True, (0, 0, 0))
    window.blit(info_text, (10, 10))

    # 3. Map Rendering with Scale 1.3
    map_start_y = 100
    state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
    
    grid_scale = 1.3 
    if grid_scale != 1.0:
        new_w = int(state_surface.get_width() * grid_scale)
        new_h = int(state_surface.get_height() * grid_scale)
        state_surface = pygame.transform.scale(state_surface, (new_w, new_h))

    surf_width, surf_height = state_surface.get_size()
    start_x = (screen_width - surf_width) // 2
    window.blit(state_surface, (start_x, map_start_y))

    grid_width, grid_height = len(env.mdp.terrain_mtx[0]), len(env.mdp.terrain_mtx)
    tile_w, tile_h = surf_width / grid_width, surf_height / grid_height

    # 4. [Speech Bubbles] Rendering (Level 1)
    if thought_msg and visual_level == 1:
        raw_lines = thought_msg.split('\n')
        
        icon_map = {
            "pickup_onion": "onions", 
            "pickup_dish": "dishes", 
            "pickup_tomato": "tomatoes",
            "put_onion_in_pot": "soup_idle_tomato_0_onion_1", 
            "put_tomato_in_pot": "pot",
            "fill_dish_with_soup": "soup_done_tomato_0_onion_3",
            "deliver_soup": "SOUTH-soup-onion",
            "wait": "stay"
        } 

        action_priorities = [
            ("put_onion_in_pot", 'P'), ("put_tomato_in_pot", 'P'), ("fill_dish_with_soup", 'P'),
            ("pickup_onion", 'O'), ("pickup_dish", 'D'), ("pickup_tomato", 'T'), ("deliver_soup", 'S'),
            ("place_obj_on_counter", 'C'), ("wait", None)
        ]

        intention_str, plan_str = None, None
        for line in raw_lines:
            line = line.strip()
            if not line: continue
            if "Plan" in line:
                m = re.search(r'Player (\d+)', line)
                if m and int(m.group(1)) == num_AI:
                    plan_str = line.split(':')[-1].strip().replace('"', '')
            elif "Intention" in line and show_intention:
                m = re.search(r'Player (\d+)', line)
                if m and int(m.group(1)) != num_AI: 
                    intention_str = line.split(':')[-1].strip().replace('"', '')

        def get_action_surf(action_str, is_thought):
            if not action_str: return None
            lower_content = action_str.lower()
            found_key = None
            for skill_key, char in action_priorities:
                if skill_key in lower_content:
                    found_key = skill_key
                    break
            icon_name = icon_map.get(found_key) if found_key else None
            temp = pygame.Surface((40, 40), pygame.SRCALPHA)
            if icon_name:
                for src in [visualizer.OBJECTS_IMG, visualizer.TERRAINS_IMG, visualizer.SOUPS_IMG, visualizer.CHEFS_IMG]:
                    try: 
                        name_to_try = icon_name if ".png" in icon_name else icon_name + ".png"
                        try:
                            src.blit_on_surface(temp, (0, 0), icon_name)
                        except:
                            src.blit_on_surface(temp, (0, 0), name_to_try)
                        break
                    except: 
                        continue
            
            icon_size = (60, 60) 
            max_img_w, max_img_h = (46, 46) 
            
            bounding_rect = temp.get_bounding_rect()
            if bounding_rect.width > 0 and bounding_rect.height > 0:
                actual_icon_surf = temp.subsurface(bounding_rect)
                orig_w, orig_h = actual_icon_surf.get_size()
                scale = min(max_img_w / orig_w, max_img_h / orig_h)
                
                final_w = int(orig_w * scale)
                final_h = int(orig_h * scale)
                final_icon_surf = pygame.transform.scale(actual_icon_surf, (final_w, final_h))
            else:
                final_w, final_h = (max_img_w, max_img_h)
                final_icon_surf = pygame.transform.scale(temp, (final_w, final_h))

            bg_surf = pygame.Surface(icon_size, pygame.SRCALPHA)
            try:
                pygame.draw.rect(bg_surf, (230, 230, 230), (0, 0, *icon_size), border_radius=8)
            except TypeError:
                pygame.draw.rect(bg_surf, (230, 230, 230), (0, 0, *icon_size))
            
            offset_x = (icon_size[0] - final_w) // 2
            offset_y = (icon_size[1] - final_h) // 2
            bg_surf.blit(final_icon_surf, (offset_x, offset_y))
            
            return bg_surf

        font_prefix = pygame.font.SysFont("arial", 20, bold=True)
        line_surfs = []

        if intention_str:
            prefix = font_prefix.render("Guess you'll: ", True, (0, 0, 0))
            act_surf = get_action_surf(intention_str, is_thought=True)
            if act_surf:
                line_w = prefix.get_width() + act_surf.get_width()
                line_h = max(prefix.get_height(), act_surf.get_height())
                line_surf = pygame.Surface((line_w, line_h), pygame.SRCALPHA)
                line_surf.blit(prefix, (0, (line_h - prefix.get_height()) // 2))
                line_surf.blit(act_surf, (prefix.get_width(), (line_h - act_surf.get_height()) // 2))
                line_surfs.append(line_surf)

        if plan_str:
            prefix = font_prefix.render("My plan: ", True, (0, 0, 0))
            act_surf = get_action_surf(plan_str, is_thought=False)
            if act_surf:
                line_w = prefix.get_width() + act_surf.get_width()
                line_h = max(prefix.get_height(), act_surf.get_height())
                line_surf = pygame.Surface((line_w, line_h), pygame.SRCALPHA)
                line_surf.blit(prefix, (0, (line_h - prefix.get_height()) // 2))
                line_surf.blit(act_surf, (prefix.get_width(), (line_h - act_surf.get_height()) // 2))
                line_surfs.append(line_surf)

        if line_surfs:
            total_w = max(s.get_width() for s in line_surfs)
            total_h = sum(s.get_height() for s in line_surfs) + (10 * (len(line_surfs) - 1)) 
            combined_surf = pygame.Surface((total_w, total_h), pygame.SRCALPHA)
            current_y = 0
            for s in line_surfs:
                x_pos = (total_w - s.get_width()) // 2
                combined_surf.blit(s, (x_pos, current_y))
                current_y += s.get_height() + 10
            
            px, py = get_player_screen_pos(num_AI, env, map_start_y, start_x, surf_width, surf_height)
            draw_speech_bubble(window, combined_surf, px, py, is_thought=False, border_color=(0, 0, 0), border_width=2, alpha=200, y_offset=110, padding=6)

    # 5. [Visual Level 2] Highlight Logic
    elif visual_level == 2 and thought_msg:
        try:
            highlight_for_inference_coords, highlight_for_plan_coords = parse_separate_highlights(thought_msg, layout_dict, num_AI=num_AI)
            if not show_intention: highlight_for_inference_coords = []
            
            # 색상 정의
            inf_color = (50, 120, 255)       # 파란색 (파트너 행동)
            plan_color = (255, 255, 255)     # 하얀색 (AI 계획)
            overlap_color = (0, 200, 255)    # 쨍한 하늘색 (겹칠 때)

            # 좌표 중복을 찾기 위해 Set으로 변환
            inf_set = set(highlight_for_inference_coords)
            plan_set = set(highlight_for_plan_coords)
            
            # 교집합(겹치는 곳)과 여집합 분리
            overlap_set = inf_set.intersection(plan_set)
            only_inf_set = inf_set - overlap_set
            only_plan_set = plan_set - overlap_set

            # 각각 할당된 색상으로 렌더링
            render_groups = [
                (only_inf_set, inf_color),
                (only_plan_set, plan_color),
                (overlap_set, overlap_color)
            ]

            for coords, color in render_groups:
                if not coords: continue
                s = pygame.Surface((int(tile_w), int(tile_h)), pygame.SRCALPHA)
                # 💡 투명도(Alpha)를 100에서 120으로 살짝 올려서 색이 더 또렷하게 보이게 조절
                s.fill((*color, 100))
                for (hx, hy) in coords:
                    dx, dy = start_x + (hx * tile_w), map_start_y + (hy * tile_h)
                    window.blit(s, (dx, dy))
                    pygame.draw.rect(window, color, pygame.Rect(dx, dy, tile_w, tile_h), 3)
        except Exception as e:
             print(f"Error in Level 2 rendering: {e}")

    pygame.display.flip()