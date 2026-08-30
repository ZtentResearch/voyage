#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
transfer_images.py

Extracts base64-encoded images from a Datalab JSON document output
and embeds them as Data URIs directly into the corresponding Markdown file.
"""

import argparse
import base64
import binascii
import contextlib
import json
import mimetypes
import os
import sys
from pathlib import Path


def get_mime_type(filename: str, b64_data: str | None = None) -> str:
    """Determine MIME type from filename extension or base64 magic bytes."""
    ext = os.path.splitext(filename)[1].lower()
    ext_mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".bmp": "image/bmp",
    }
    if ext in ext_mime_map:
        return ext_mime_map[ext]

    guessed_type, _ = mimetypes.guess_type(filename)
    if guessed_type:
        return guessed_type

    if b64_data:
        with contextlib.suppress(binascii.Error, ValueError):
            sample = base64.b64decode(b64_data[:64])
            if sample.startswith(b"\xff\xd8\xff"):
                return "image/jpeg"
            if sample.startswith(b"\x89PNG\r\n\x1a\n"):
                return "image/png"
            if sample.startswith((b"GIF87a", b"GIF89a")):
                return "image/gif"
            if sample.startswith(b"RIFF") and sample[8:12] == b"WEBP":
                return "image/webp"

    return "image/jpeg"


def extract_images_from_json(data: object, images: dict[str, str] | None = None) -> dict[str, str]:
    """Recursively search for 'images' dictionaries across all nested JSON nodes."""
    if images is None:
        images = {}

    if isinstance(data, dict):
        if "images" in data and isinstance(data["images"], dict):
            for img_name, img_b64 in data["images"].items():
                if isinstance(img_b64, str) and img_b64.strip():
                    images[img_name] = img_b64.strip()

        for value in data.values():
            extract_images_from_json(value, images)
    elif isinstance(data, list):
        for item in data:
            extract_images_from_json(item, images)

    return images


def transfer_images_to_markdown(
    json_path: Path | str,
    md_path: Path | str,
    output_path: Path | str | None = None,
) -> tuple[int, int]:
    """
    Reads images from JSON, embeds them into Markdown as data URIs,
    and writes the updated markdown to output_path (or md_path if output_path is None).
    
    Returns:
        (total_images_found, total_replacements_made)
    """
    json_file = Path(json_path)
    md_file = Path(md_path)
    out_file = Path(output_path) if output_path else md_file

    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    if not md_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_file}")

    print(f"Reading JSON: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    images = extract_images_from_json(json_data)
    print(f"Found {len(images)} base64 images in JSON.")

    print(f"Reading Markdown: {md_file}")
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    total_replacements = 0
    updated_md = md_content

    for img_name, b64_data in images.items():
        mime_type = get_mime_type(img_name, b64_data)
        data_uri = f"data:{mime_type};base64,{b64_data}"
        count = updated_md.count(img_name)
        if count > 0:
            updated_md = updated_md.replace(img_name, data_uri)
            total_replacements += count
            print(f"  - Replaced {count} occurrence(s) of '{img_name}' ({mime_type})")
        elif data_uri in updated_md:
            print(f"  - '{img_name}' is already embedded as a Data URI.")
        else:
            print(f"  - Notice: '{img_name}' not referenced in markdown.")

    print(f"Writing updated Markdown to: {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(updated_md)

    orig_size_kb = len(md_content.encode("utf-8")) / 1024
    new_size_kb = len(updated_md.encode("utf-8")) / 1024
    print(f"Done! Markdown size: {orig_size_kb:.1f} KB -> {new_size_kb:.1f} KB")

    return len(images), total_replacements


def main():
    default_dir = Path(__file__).resolve().parent
    default_json = default_dir / "datalab-output-Deep RL Hands-On 3E - pages 140-170.pdf.json"
    default_md = default_dir / "datalab-output-Deep RL Hands-On 3E - pages 140-170.pdf.md"

    parser = argparse.ArgumentParser(
        description="Transfer base64 images from Datalab JSON to Markdown file."
    )
    parser.add_argument(
        "--json",
        "-j",
        type=str,
        default=str(default_json),
        help=f"Path to input JSON file (default: {default_json.name})",
    )
    parser.add_argument(
        "--md",
        "-m",
        type=str,
        default=str(default_md),
        help=f"Path to input Markdown file (default: {default_md.name})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output Markdown file (default: in-place overwrite of input markdown)",
    )
    parser.add_argument(
        "--in-place",
        "-i",
        action="store_true",
        help="Explicitly overwrite the input Markdown file in-place",
    )

    args = parser.parse_args()

    out_path = args.output
    if out_path is None and not args.in_place:
        # Default behavior is to update in-place unless specified otherwise
        out_path = args.md

    try:
        transfer_images_to_markdown(args.json, args.md, out_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
