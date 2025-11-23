#!/usr/bin/env python3
"""
Session Inspector - Analyze and report on all saved sessions

Usage:
    python scripts/inspect_sessions.py              # List all sessions
    python scripts/inspect_sessions.py --latest     # Show latest session details
    python scripts/inspect_sessions.py --all        # Show all session details
    python scripts/inspect_sessions.py <session_id> # Show specific session
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"


def format_size(bytes_size: int) -> str:
    """Format file size to human-readable string."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.1f} GB"


def get_session_info(session_dir: Path) -> dict:
    """Extract information about a session."""
    info = {
        "session_id": session_dir.name,
        "path": str(session_dir),
        "created": datetime.fromtimestamp(session_dir.stat().st_ctime),
        "config": None,
        "chunk_count": 0,
        "has_combined": False,
        "metadata": None,
        "total_size": 0,
    }
    
    # Load config
    config_files = list(session_dir.glob("session_config_*.json"))
    if config_files:
        with open(config_files[0], 'r', encoding='utf-8') as f:
            info["config"] = json.load(f)
    
    # Count chunks
    chunk_files = list(session_dir.glob("chunk_*.wav"))
    info["chunk_count"] = len(chunk_files)
    
    # Check for combined audio
    combined_path = session_dir / "combined_audio.wav"
    if combined_path.exists():
        info["has_combined"] = True
        info["combined_size"] = combined_path.stat().st_size
    
    # Load metadata
    metadata_path = session_dir / "audio_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            info["metadata"] = json.load(f)
    
    # Calculate total size
    info["total_size"] = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())
    
    return info


def print_session_summary(info: dict, detailed: bool = False):
    """Print a summary of a session."""
    session_id = info["session_id"]
    config = info.get("config", {})
    cfg = config.get("config", {}) if config else {}
    
    print(f"\n{'=' * 80}")
    print(f"Session: {session_id}")
    print(f"{'=' * 80}")
    
    print(f"\nCreated: {info['created'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    if config:
        print(f"\nConfiguration:")
        print(f"  Surah: {cfg.get('surah', 'N/A')}")
        print(f"  Ayah: {cfg.get('ayah', 'N/A')}")
        print(f"  Start Word: {cfg.get('start_word', 1)}")
        print(f"  Num Words: {cfg.get('num_words', 'default')}")
        print(f"  Rewaya: {cfg.get('rewaya', 'hafs')}")
        print(f"  Chunk Duration: {cfg.get('chunk_duration', 2000)} ms")
        print(f"  Max Chunks: {cfg.get('max_chunks', 5)}")
    
    print(f"\nAudio Data:")
    print(f"  Chunks Received: {info['chunk_count']}")
    
    if info["has_combined"]:
        print(f"  Combined Audio: ✓ ({format_size(info['combined_size'])})")
        
        if info["metadata"]:
            meta = info["metadata"]
            print(f"  Duration: {format_duration(meta.get('duration_seconds', 0))}")
            print(f"  Sample Rate: {meta.get('sample_rate', 0)} Hz")
            print(f"  Total Samples: {meta.get('total_samples', 0):,}")
    else:
        print(f"  Combined Audio: ✗ (session incomplete)")
    
    print(f"\nStorage:")
    print(f"  Total Size: {format_size(info['total_size'])}")
    print(f"  Location: {info['path']}")
    
    if detailed:
        print(f"\nFiles:")
        session_dir = Path(info["path"])
        for file in sorted(session_dir.iterdir()):
            if file.is_file():
                size = format_size(file.stat().st_size)
                print(f"  - {file.name:40s} {size:>10s}")


def list_sessions():
    """List all sessions with basic info."""
    sessions_dir = Path("sessions")
    
    if not sessions_dir.exists():
        print("No sessions directory found.")
        return []
    
    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
    
    if not session_dirs:
        print("No sessions found.")
        return []
    
    # Sort by creation time (newest first)
    session_dirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)
    
    print(f"\nFound {len(session_dirs)} session(s):\n")
    print(f"{'Session ID':<40} {'Created':<20} {'Chunks':<10} {'Size':<10} {'Status'}")
    print("-" * 100)
    
    sessions_info = []
    for session_dir in session_dirs:
        info = get_session_info(session_dir)
        sessions_info.append(info)
        
        status = "✓ Complete" if info["has_combined"] else "⧗ Incomplete"
        created = info["created"].strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"{info['session_id']:<40} {created:<20} {info['chunk_count']:<10} "
              f"{format_size(info['total_size']):<10} {status}")
    
    return sessions_info


def main():
    parser = argparse.ArgumentParser(description="Inspect saved WebSocket sessions")
    parser.add_argument("session_id", nargs="?", help="Specific session ID to inspect")
    parser.add_argument("--latest", action="store_true", help="Show details of latest session")
    parser.add_argument("--all", action="store_true", help="Show details of all sessions")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed file listing")
    
    args = parser.parse_args()
    
    sessions_dir = Path("sessions")
    
    if not sessions_dir.exists():
        print("❌ No sessions directory found.")
        print("\nHint: Sessions are created when WebSocket connections are established.")
        return 1
    
    # List all sessions
    sessions_info = list_sessions()
    
    if not sessions_info:
        return 1
    
    # Show specific session
    if args.session_id:
        session_path = sessions_dir / args.session_id
        if not session_path.exists():
            print(f"\n❌ Session '{args.session_id}' not found.")
            return 1
        
        info = get_session_info(session_path)
        print_session_summary(info, detailed=args.detailed or True)
    
    # Show latest session
    elif args.latest:
        info = sessions_info[0]  # Already sorted by date
        print_session_summary(info, detailed=args.detailed or True)
    
    # Show all sessions
    elif args.all:
        for info in sessions_info:
            print_session_summary(info, detailed=args.detailed)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
