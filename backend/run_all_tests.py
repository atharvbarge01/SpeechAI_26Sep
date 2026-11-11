import subprocess
import sys

# Set stdout encoding to utf-8 if possible
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import os

backend_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(backend_dir, "tests")

def run_script(name):
    print("=" * 60)
    print(f"RUNNING TEST SUITE: {name}")
    print("=" * 60)
    test_path = os.path.join(tests_dir, name)
    env = os.environ.copy()
    env["PYTHONPATH"] = backend_dir
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run([sys.executable, test_path], capture_output=True, text=True, encoding='utf-8', cwd=backend_dir, env=env)
    print(res.stdout)
    if res.stderr:
        print("Error details:")
        print(res.stderr)
    return res.returncode

def main():
    test_files = [
        "test_streak_logic.py",
        "test_fluency_analytics.py",
        "test_auth_flow.py",
        "test_groq_generation.py",
        "test_pronunciation_alignment.py",
        "test_speech_pathology_analysis.py",
        "test_feedback_evaluation.py",
        "test_inhouse_ml_model.py"
    ]
    
    failed = False
    for tf in test_files:
        code = run_script(tf)
        if code != 0:
            print(f"[FAIL] TEST SUITE {tf} FAILED!")
            failed = True
        else:
            print(f"[SUCCESS] TEST SUITE {tf} PASSED SUCCESSFULLY!")
        print("\n")
        
    if failed:
        print("=" * 60)
        print("[FAIL] SOME TEST SUITES FAILED. PLEASE RESOLVE.")
        print("=" * 60)
        sys.exit(1)
    else:
        print("=" * 60)
        print("[SUCCESS] ALL SPEECH_AI TEST SUITES PASSED METICULOUSLY!")
        print("=" * 60)

if __name__ == "__main__":
    main()
