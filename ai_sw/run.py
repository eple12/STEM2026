#!/usr/bin/env python
"""Convenience launcher so you don't have to remember `python -m game`.

    python run.py                 # Monza, 3 laps
    python run.py --track Spa
    python run.py --track Silverstone --laps 5 --fullscreen
"""
from game.app import main

if __name__ == "__main__":
    main()
