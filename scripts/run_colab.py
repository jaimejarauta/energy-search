#!/usr/bin/env python3
"""Single-Command Self-Healing Google Colab Runner for Energy-Search.
Handles 60s timeout credit protection, multi-key W&B authentication,
calibrated benchmark execution, and automatic VM teardown.
"""

import os
import sys
import signal
import subprocess

def main():
    print("=" * 65)
    print("ENERGY-SEARCH: STAGE 2 GSM8K AUTOMATED COLAB BENCHMARK")
    print("=" * 65)

    # 1. Credit Guard: Strict 60-second timeout for any interactive prompt
    def timeout_handler(signum, frame):
        print("\n[Credit Guard Alert] Interactive auth timed out (60s without input)!")
        print("[Credit Guard Alert] Unassigning Colab GPU immediately to prevent idle burn...")
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception:
            pass
        sys.exit(1)

    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

    # 2. Multi-Candidate Colab Secrets Search
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        try:
            from google.colab import userdata
            candidate_keys = [
                'WANDB_API_KEY', 'wandb_api_key', 'WANDB_KEY', 'wandb_key',
                'wandb', 'WANDB', 'wandb_token', 'WANDB_TOKEN', 'API_KEY', 'api_key'
            ]
            for k in candidate_keys:
                try:
                    val = userdata.get(k)
                    if val and len(val.strip()) >= 20:
                        wandb_key = val.strip()
                        print(f"[W&B Guard] Successfully loaded API key from Colab Secret: '{k}'")
                        break
                except Exception:
                    continue
        except Exception:
            pass

    # Check ~/.netrc as fallback
    netrc_path = os.path.expanduser('~/.netrc')
    if not wandb_key and os.path.exists(netrc_path):
        try:
            with open(netrc_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0] == 'password' and len(parts[1]) >= 20:
                        wandb_key = parts[1]
                        print("[W&B Guard] Restored API key from ~/.netrc")
                        break
        except Exception:
            pass

    # 3. Authenticate or Fallback to Offline Mode
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
        os.environ['WANDB_MODE'] = 'online'
        try:
            import wandb
            wandb.login(key=wandb_key, relogin=False)
            print("[W&B Guard] Authenticated in online mode.")
        except Exception as e:
            print(f"[W&B Guard Warning] Login warning: {e}. Will sync after completion.")
    else:
        print("[W&B Guard Notice] No W&B key detected. Defaulting to offline mode (never hangs).")
        os.environ['WANDB_MODE'] = 'offline'

    # Clear interactive timeout alarm now that auth is resolved
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(0)

    # 4. Install Dependencies
    print("\n[Runner] Ensuring required Python dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers", "accelerate", "datasets", "wandb"
    ], check=True)

    # 5. Execute Calibrated Stage 2 Benchmark
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    eval_script = os.path.join(repo_root, "scripts", "eval_gsm8k.py")
    wandb_mode = os.environ.get("WANDB_MODE", "online")

    cmd = [
        sys.executable, eval_script,
        "--model-name", "Qwen/Qwen2.5-Math-1.5B",
        "--num-samples", "50",
        "--tau", "1.5",
        "--k-persist", "3",
        "--freeze-vel-tau", "0.25",
        "--freeze-ent-tau", "0.80",
        "--k-freeze", "3",
        "--colinear-tau", "0.85",
        "--wandb",
        "--wandb-project", "research-projekts",
        "--wandb-mode", wandb_mode,
        "--wandb-name", "energy-search-1.5b-calibrated-l4"
    ]

    print(f"\n[Runner] Executing: {' '.join(cmd)}\n")
    ret = subprocess.run(cmd, env=os.environ.copy())

    # 6. Flush & Sync Offline W&B Runs
    print("\n[Runner] Syncing all W&B runs to cloud...")
    subprocess.run("wandb sync --include-offline $(find /tmp/wandb ./wandb -type d -name 'offline-run-*' 2>/dev/null) 2>/dev/null || true", shell=True)

    # 7. Credit Guard: Teardown VM
    try:
        from google.colab import runtime
        print("\n[Credit Guard] Run complete. Unassigning GPU to save Colab credits...")
        runtime.unassign()
    except Exception as e:
        print(f"[Credit Guard] Teardown notice: {e}")

    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()
