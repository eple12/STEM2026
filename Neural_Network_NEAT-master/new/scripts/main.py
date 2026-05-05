from pathlib import Path
import subprocess
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1] / "neat_racetrack_viz"
    cmd = [sys.executable, str(project_root / "main.py")]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
