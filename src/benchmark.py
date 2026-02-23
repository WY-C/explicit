import os
import time
import json
import numpy as np
import datetime
import pygame  # Pygame 추가
import visualization_utils as vu # 시각화 모듈 추가
from argparse import ArgumentParser
from distutils.util import strtobool
import warnings
import sys

# 불필요한 경고 차단
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Rich 라이브러리
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

# Overcooked AI
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer # 시각화 도구
from overcooked_ai_py.agents.agent import AgentGroup
from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent
import importlib_metadata

try:
    VERSION = importlib_metadata.version("overcooked_ai")
except:
    VERSION = "0.0.1"

def boolean_argument(value):
    return bool(strtobool(value))

def get_combined_thought(agents_list):
    for i, agent in enumerate(agents_list):
        if hasattr(agent, 'current_thought') and agent.current_thought:
            return i, agent.current_thought
    return -1, None

def run_benchmark(args):
    console = Console()
    
    # 1. 설정
    target_agent_name = args.target_agent
    layouts = args.layouts 
    opponents = ['SP', 'PBT', 'MEP', 'FCP', 'COLE'] # 벤치마크 대상
    num_episodes = 5
    horizon = args.horizon
    visual_level = args.visual_level
    
    benchmark_results = { "config": vars(args), "results": {} }

    console.print(f"\n[bold green]🚀 Visual Benchmark Started: {target_agent_name}[/bold green]")
    console.print(f"Layouts: {layouts} | Opponents: {opponents}")

    # 2. Pygame 초기화 (한 번만)
    pygame.init()
    # 화면 크기 설정 (필요시 조정)
    window_surface = pygame.display.set_mode((900, 600))
    pygame.display.set_caption(f"Benchmark: {target_agent_name}")
    visualizer = StateVisualizer(cook_time=20) # cook_time은 기본값

    total_steps = len(layouts) * len(opponents) * num_episodes

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn(" | Last Score: [bold cyan]{task.fields[last_score]}"),
    ) as progress:
        
        main_task = progress.add_task("[green]Running...", total=total_steps, last_score="N/A")

        for layout in layouts:
            benchmark_results["results"][layout] = {}
            
            # 맵 로드
            mdp_layout = NEW_LAYOUTS.get(layout, layout) if VERSION == '1.1.0' else OLD_LAYOUTS.get(layout, layout)
            mdp = OvercookedGridworld.from_layout_name(mdp_layout)
            env = OvercookedEnv(mdp, horizon=horizon)
            layout_dict = vu.generate_layout_dict(mdp) # 시각화용 레이아웃 데이터

            for opponent_name in opponents:
                scores = []
                
                for ep in range(num_episodes):
                    progress.update(main_task, description=f"[{layout}] vs {opponent_name} ({ep+1}/{num_episodes})")
                    
                    try:
                        # 에이전트 생성
                        agent0 = make_agent(target_agent_name, mdp, layout, 
                                          model=args.gpt_model, prompt_level=args.prompt_level,
                                          belief_revision=args.belief_revision, 
                                          retrival_method=args.retrival_method, K=args.K)
                        agent1 = make_agent(opponent_name, mdp, layout, seed_id=ep)

                        agents_list = [agent0, agent1]
                        team = AgentGroup(*agents_list)
                        team.reset()
                        env.reset()
                        
                        total_reward = 0
                        done = False
                        
                        # [WarmUp] 첫 실행 딜레이 방지
                        if hasattr(agent0, 'generate_ml_action'):
                             vu.draw_centered_text(window_surface, f"Loading {opponent_name}...", "Thinking...", color=(0, 0, 255))
                             _ = agent0.generate_ml_action(env.state)

                        # === 게임 루프 (딜레이 없음 + 렌더링 포함) ===
                        for t in range(horizon):
                            # 이벤트 처리 (창 닫기 등 방지)
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit(); return

                            # 1. 행동 결정 (LLM 추론 시간 소요)
                            s_t = env.state
                            a_t = team.joint_action(s_t)
                            
                            # 2. 환경 업데이트
                            obs, reward, done, info = env.step(a_t)
                            total_reward += reward

                            # 3. 화면 갱신 (인위적 딜레이 없음)
                            thought_idx, thought_msg = get_combined_thought(agents_list)
                            
                            # 제목 표시줄에 현재 상태 표시
                            pygame.display.set_caption(f"[{layout}] vs {opponent_name} (Ep {ep+1}) | Step: {t}/{horizon} | Score: {total_reward}")
                            
                            vu.render_game(
                                window=window_surface, visualizer=visualizer, env=env, 
                                step=t, horizon=horizon, reward=total_reward, 
                                num_AI=thought_idx, 
                                visual_level=visual_level, layout_dict=layout_dict, 
                                thought_msg=thought_msg, show_intention=True
                            )
                            
                            if done: break
                        # ==========================================
                        
                        scores.append(total_reward)
                        progress.update(main_task, advance=1, last_score=str(total_reward))

                    except Exception as e:
                        console.print(f"[bold red]Error in {layout} vs {opponent_name}: {e}[/bold red]")
                        scores.append(0)
                        progress.update(main_task, advance=1)
                
                # 통계 저장
                benchmark_results["results"][layout][opponent_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "scores": [float(s) for s in scores]
                }

    # 3. 종료 및 저장
    pygame.quit() # Pygame 종료

    console.print("\n[bold]📊 Benchmark Summary[/bold]")
    for layout in layouts:
        table = Table(title=f"Results: {layout}")
        table.add_column("Opponent", style="cyan"); table.add_column("Mean", style="magenta"); table.add_column("Std", style="green")
        for opp in opponents:
            data = benchmark_results["results"][layout].get(opp, {})
            if data: table.add_row(opp, f"{data['mean']:.1f}", f"{data['std']:.1f}")
        console.print(table); console.print("\n")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.gpt_model.replace('/', '-')
    filename = f"visual_bench_{target_agent_name}_{safe_model}_{timestamp}.json"
    with open(filename, "w") as f: json.dump(benchmark_results, f, indent=4)
    console.print(f"[bold blue]💾 Saved: {filename}[/bold blue]")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--target_agent', type=str, default='ProAgent')
    parser.add_argument('--layouts', nargs='+', default=['cramped_room'])
    parser.add_argument('--horizon', type=int, default=400)
    
    # 시각화 레벨 (0: 맵만, 1: 아이콘, 2: 텍스트)
    parser.add_argument('--visual_level', type=int, default=1, help='0:Map, 1:Emoji, 2:Text')

    # LLM 설정
    parser.add_argument('--gpt_model', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--prompt_level', type=str, default='l3-aip')
    parser.add_argument('--belief_revision', type=boolean_argument, default=False)
    parser.add_argument('--retrival_method', type=str, default="recent_k")
    parser.add_argument('--K', type=int, default=1)

    args = parser.parse_args()
    run_benchmark(args)