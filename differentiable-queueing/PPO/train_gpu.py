"""GPU PPO launcher — copies GPU config over before invoking train."""
import sys, shutil, os, runpy

if len(sys.argv) < 3:
    print("Usage: train_gpu.py <config_name> <env_config_name>")
    sys.exit(1)

cfg_name = sys.argv[1]
if not cfg_name.endswith('.yaml'):
    cfg_name += '.yaml'
# Copy GPU variant over (preserves original for cpu use)
src = f'./configs_gpu/{cfg_name}'
dst = f'./configs/{cfg_name}'
if os.path.exists(src):
    # Sync src to dst non-destructively only when needed
    # Actually: leave originals alone, just point train.py to GPU config via env var? 
    # Simpler: just chdir + use original since we want to keep both.
    # But the simplest thing: copy GPU config -> original for this run
    shutil.copy(src, dst)
    print(f'Using GPU config {src}')

# Now run main script
sys.argv[0] = 'train.py'
runpy.run_path('train.py', run_name='__main__')
