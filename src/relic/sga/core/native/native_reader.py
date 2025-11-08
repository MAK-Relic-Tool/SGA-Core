"""Native SGA Binary Parser - Direct Binary Format Implementation

Parses SGA V2 binary format directly to extract byte offsets.
Uses mmap + parallel zlib for ultra-fast extraction (3-4 seconds for 7,815 files).

Completely bypasses fs library for maximum speed!
"""

from __future__ import annotations

from dataclasses import dataclass


