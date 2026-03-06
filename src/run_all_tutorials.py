import subprocess
import sys

def run_step(multi, is_async, visual_level, desc):
    print(f"\n>>> {desc} 시작")
    cmd = [
        sys.executable, "tutorial.py",
        "--multi", str(multi),
        "--async", str(is_async),
        "--visual_level", str(visual_level)
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    # 1. 싱글 (비실시간 -> 실시간)
    run_step(False, False, 0, "Step 1-1: 싱글 비실시간")
    run_step(False, True, 0, "Step 1-2: 싱글 실시간")

    # 2. 멀티 비실시간 (Highlight -> NL)
    run_step(True, False, 2, "Step 2-1: 멀티 비실시간 (Highlight)")
    run_step(True, False, 1, "Step 2-2: 멀티 비실시간 (NL)")

    # 3. 멀티 실시간 (Highlight -> NL)
    run_step(True, True, 2, "Step 3-1: 멀티 실시간 (Highlight)")
    run_step(True, True, 1, "Step 3-2: 멀티 실시간 (NL)")

    print("\n모든 튜토리얼이 완료되었습니다!")