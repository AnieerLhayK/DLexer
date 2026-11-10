"""Shared pytest configuration.

GPU tests skip themselves when CUDA is unavailable. Network tests are opt-in
through ``DLEXER_NETWORK_TESTS=1`` so the default suite stays offline-friendly.
"""
