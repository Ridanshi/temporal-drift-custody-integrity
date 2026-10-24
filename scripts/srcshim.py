"""Degraded-source mode (SRC=youtube): the examiner holds the YouTube copy instead of the native file.
Genuine edge = YouTube copy -> WhatsApp copy; forged edges = YouTube copy -> forged file."""
import os


def apply(T):
    if os.environ.get("SRC") == "youtube":
        for t in T:
            t["native"], t["youtube"] = t["youtube"], t["whatsapp"]
    return T
