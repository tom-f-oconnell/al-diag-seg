"""Fake flies as a database export, for testing: see `alseg.synthetic`.

Usage: python make_synthetic.py OUT [N_FLIES]
"""
from alseg.synthetic import *  # noqa: F401,F403 (smoke_test.py uses its constants)
from alseg.synthetic import main

if __name__ == "__main__":
    main()
