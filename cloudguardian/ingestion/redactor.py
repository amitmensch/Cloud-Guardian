"""
Log Redaction & Normalization Engine for CloudGuardian.
Normalizes variable and sensitive tokens (<IP>, <NUM>, <PATH>, <HEX>, <BLOCK_ID>)
to prevent template explosion in the Drain3 parser.
"""

import re
from typing import Dict, Any, Tuple, Optional


class LogRedactor:
    """
    Applies configurable regex patterns to scrub dynamic tokens from raw logs,
    while optionally extracting session identifiers (such as HDFS block IDs).
    """

    def __init__(self, extract_hdfs_block: bool = True):
        self.extract_hdfs_block = extract_hdfs_block

        # Precompiled regex patterns (ordered by specificity to prevent partial masking)
        self.patterns = [
            # 1. HDFS Block ID (e.g., blk_-1608999687919862906 or blk_358750814)
            (re.compile(r"blk_-?\d+"), "<BLOCK_ID>"),
            # 2. Hexadecimal addresses (e.g., 0x7fff5fbff8b0 or 0xdeadbeef)
            (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
            # 3. IPv4 and IPv6 addresses with optional port (e.g., 10.251.43.115:50010)
            (
                re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"),
                "<IP>",
            ),
            # 4. Absolute Unix and Windows file paths
            (
                re.compile(r"(?:/[a-zA-Z0-9_\.\-]+)+/(?:[a-zA-Z0-9_\.\-]+)"),
                "<PATH>",
            ),
            (
                re.compile(r"[a-zA-Z]:\\(?:[a-zA-Z0-9_\.\-]+\\)+[a-zA-Z0-9_\.\-]+"),
                "<PATH>",
            ),
            # 5. UUIDs / GUIDs
            (
                re.compile(
                    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
                ),
                "<UUID>",
            ),
            # 6. Process IDs / Thread IDs formatted as [pid:1234] or pid=1234
            (re.compile(r"(?:pid|tid|thread)[=: ]\d+", re.IGNORECASE), "<PID>"),
            # 7. Isolated numbers / ports / counters
            (re.compile(r"\b\d+\b"), "<NUM>"),
        ]

        # Extractors for common metadata
        self._block_id_re = re.compile(r"(blk_-?\d+)")
        self._level_re = re.compile(
            r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b",
            re.IGNORECASE,
        )
        self._hdfs_timestamp_re = re.compile(r"^(\d{6} \d{6})")

    def extract_metadata(self, raw_line: str) -> Dict[str, Any]:
        """Extracts contextual metadata such as block ID, severity level, and timestamps."""
        metadata: Dict[str, Any] = {
            "block_id": None,
            "level": "INFO",
            "timestamp_str": None,
        }

        # Block ID extraction
        block_match = self._block_id_re.search(raw_line)
        if block_match:
            metadata["block_id"] = block_match.group(1)

        # Log Level extraction
        level_match = self._level_re.search(raw_line)
        if level_match:
            metadata["level"] = level_match.group(1).upper()

        # Timestamp extraction (e.g. 081109 203518 in HDFS)
        ts_match = self._hdfs_timestamp_re.match(raw_line)
        if ts_match:
            metadata["timestamp_str"] = ts_match.group(1)

        return metadata

    def redact(self, raw_line: str) -> str:
        """Applies regex normalization to replace variable tokens with standard placeholders."""
        cleaned = raw_line.strip()
        for pattern, placeholder in self.patterns:
            cleaned = pattern.sub(placeholder, cleaned)
        # Collapse repeated whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def process(self, raw_line: str) -> Tuple[str, Dict[str, Any]]:
        """Convenience method returning both the redacted line and extracted metadata."""
        meta = self.extract_metadata(raw_line)
        redacted = self.redact(raw_line)
        return redacted, meta
