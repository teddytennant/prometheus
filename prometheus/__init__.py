"""Prometheus Python package root.

Layout is spec/17 at the repo root (`model/`, `train/`, ...). This namespace
exists so F4 job templates can `from prometheus.verify.v1_parity import run_v1`
with `sys.path` pointing at the repo root.
"""
