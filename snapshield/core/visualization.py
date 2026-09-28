"""Visualization layer for highlighting sensitive detected regions on images.

Features:
- Correlates OCR token bounding boxes with hybrid detection findings.
- Renders color-coded bounding highlights based on severity and policy action:
  - CRITICAL / BLOCK: Crimson/Red
  - HIGH / REDACT: Orange
  - MEDIUM / REVIEW: Amber/Yellow
  - LOW / ALLOW: Emerald/Green
- Overlays clean badge headers (e.g. '[BLOCK: API_KEY]', '[REDACT: AADHAAR]').
- Never displays or leaks raw sensitive data onto the visual overlay.
- Exports standard PNG or JPEG image bytes.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

from snapshield.core.logger import logger
from snapshield.core.models import Detection, PolicyAction, PolicyDecision, SeverityLevel
from snapshield.core.ocr import OCRBoundingBox

# Severity / Action Color Palettes (RGBA with opacity for overlay)
COLOR_PALETTE = {
    "BLOCK": {"border": (239, 68, 68, 255), "fill": (239, 68, 68, 60), "badge": (185, 28, 28, 240)},
    "CRITICAL": {"border": (239, 68, 68, 255), "fill": (239, 68, 68, 60), "badge": (185, 28, 28, 240)},
    "REDACT": {"border": (249, 115, 22, 255), "fill": (249, 115, 22, 60), "badge": (194, 65, 12, 240)},
    "HIGH": {"border": (249, 115, 22, 255), "fill": (249, 115, 22, 60), "badge": (194, 65, 12, 240)},
    "REVIEW": {"border": (234, 179, 8, 255), "fill": (234, 179, 8, 60), "badge": (161, 98, 7, 240)},
    "MEDIUM": {"border": (234, 179, 8, 255), "fill": (234, 179, 8, 60), "badge": (161, 98, 7, 240)},
    "ALLOW": {"border": (16, 185, 129, 255), "fill": (16, 185, 129, 50), "badge": (4, 120, 87, 240)},
    "LOW": {"border": (16, 185, 129, 255), "fill": (16, 185, 129, 50), "badge": (4, 120, 87, 240)},
}


def _match_detections_to_boxes(
    extracted_text: str,
    bounding_boxes: List[Union[OCRBoundingBox, Dict[str, Any]]],
    detections: List[Detection],
) -> List[Tuple[Detection, List[Dict[str, float]]]]:
    """Correlates character-span detections with OCR word bounding boxes."""
    if not bounding_boxes or not detections:
        return []

    # Normalize boxes to dictionaries
    normalized_boxes: List[Dict[str, Any]] = []
    for b in bounding_boxes:
        if isinstance(b, OCRBoundingBox):
            normalized_boxes.append({
                "text": b.text,
                "x0": b.x0,
                "top": b.top,
                "x1": b.x1,
                "bottom": b.bottom,
            })
        elif isinstance(b, dict):
            normalized_boxes.append(b)

    # Compute approximate character spans for each bounding box in extracted_text
    # by matching tokens in sequential order
    current_search_idx = 0
    box_char_spans: List[Tuple[int, int]] = []

    for b in normalized_boxes:
        word = b.get("text", "")
        if not word:
            box_char_spans.append((-1, -1))
            continue

        pos = extracted_text.find(word, current_search_idx)
        if pos != -1:
            box_char_spans.append((pos, pos + len(word)))
            current_search_idx = pos + len(word)
        else:
            # Fallback search from start if sequential order drifted
            fallback_pos = extracted_text.find(word)
            if fallback_pos != -1:
                box_char_spans.append((fallback_pos, fallback_pos + len(word)))
            else:
                box_char_spans.append((-1, -1))

    matched_pairs: List[Tuple[Detection, List[Dict[str, float]]]] = []

    for det in detections:
        det_boxes: List[Dict[str, float]] = []
        det_start = det.start
        det_end = det.end

        for idx, (b_start, b_end) in enumerate(box_char_spans):
            if b_start == -1 or b_end == -1:
                continue

            # Check overlap between detection span and box span
            if max(det_start, b_start) < min(det_end, b_end):
                b = normalized_boxes[idx]
                det_boxes.append({
                    "x0": float(b.get("x0", 0.0)),
                    "top": float(b.get("top", 0.0)),
                    "x1": float(b.get("x1", 0.0)),
                    "bottom": float(b.get("bottom", 0.0)),
                })

        # Fallback by raw_value word matching if offset alignment found no boxes
        if not det_boxes and det.raw_value:
            raw_tokens = det.raw_value.split()
            for b in normalized_boxes:
                b_text = b.get("text", "")
                if any(t.lower() in b_text.lower() or b_text.lower() in t.lower() for t in raw_tokens if len(t) >= 2):
                    det_boxes.append({
                        "x0": float(b.get("x0", 0.0)),
                        "top": float(b.get("top", 0.0)),
                        "x1": float(b.get("x1", 0.0)),
                        "bottom": float(b.get("bottom", 0.0)),
                    })

        if det_boxes:
            matched_pairs.append((det, det_boxes))

    return matched_pairs


def annotate_image_with_detections(
    image_input: Union[Image.Image, bytes],
    extracted_text: str,
    bounding_boxes: List[Union[OCRBoundingBox, Dict[str, Any]]],
    detections: List[Detection],
    decisions: Optional[List[PolicyDecision]] = None,
    export_format: str = "PNG",
) -> Tuple[bytes, int]:
    """Renders a visualization overlay highlighting detected sensitive regions on the image.

    Args:
        image_input: Original PIL Image or image byte buffer.
        extracted_text: Text extracted from OCR.
        bounding_boxes: List of OCR bounding box items.
        detections: List of Detection domain entities.
        decisions: Optional list of PolicyDecision objects to color by action.
        export_format: Output format ('PNG' or 'JPEG').

    Returns:
        Tuple of (annotated_image_bytes, number_of_highlighted_regions).
    """
    if isinstance(image_input, bytes):
        base_img = Image.open(io.BytesIO(image_input)).convert("RGBA")
    else:
        base_img = image_input.convert("RGBA")

    # Match detections to bounding boxes
    matches = _match_detections_to_boxes(extracted_text, bounding_boxes, detections)
    if not matches:
        # Return clean image bytes if no regions matched
        out_buf = io.BytesIO()
        save_img = base_img.convert("RGB") if export_format.upper() in ("JPG", "JPEG") else base_img
        save_img.save(out_buf, format=export_format)
        return out_buf.getvalue(), 0

    # Decision map lookup by entity type
    action_map: Dict[str, PolicyAction] = {}
    if decisions:
        for dec in decisions:
            action_map[dec.entity_type] = dec.action

    # Create transparent overlay layer for highlights
    overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(base_img)

    # Standard font setup
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    highlight_count = 0

    for det, boxes in matches:
        if not boxes:
            continue

        highlight_count += 1

        # Determine color styling
        action = action_map.get(det.entity_type)
        if action == PolicyAction.BLOCK:
            style = COLOR_PALETTE["BLOCK"]
            action_label = "BLOCK"
        elif action == PolicyAction.REDACT:
            style = COLOR_PALETTE["REDACT"]
            action_label = "REDACT"
        elif action == PolicyAction.REVIEW:
            style = COLOR_PALETTE["REVIEW"]
            action_label = "REVIEW"
        elif action == PolicyAction.ALLOW:
            style = COLOR_PALETTE["ALLOW"]
            action_label = "ALLOW"
        else:
            sev_key = det.severity.value if isinstance(det.severity, SeverityLevel) else str(det.severity)
            style = COLOR_PALETTE.get(sev_key, COLOR_PALETTE["HIGH"])
            action_label = sev_key

        # Calculate union bounding rectangle for the detection
        min_x0 = min(b["x0"] for b in boxes)
        min_top = min(b["top"] for b in boxes)
        max_x1 = max(b["x1"] for b in boxes)
        max_bottom = max(b["bottom"] for b in boxes)

        # Add modest padding
        pad = 3
        box_coords = [
            max(0, min_x0 - pad),
            max(0, min_top - pad),
            min(base_img.width, max_x1 + pad),
            min(base_img.height, max_bottom + pad),
        ]

        # Draw semi-transparent fill
        draw_overlay.rectangle(box_coords, fill=style["fill"])

        # Draw crisp border on base image
        draw_base.rectangle(box_coords, outline=style["border"], width=2)

        # Draw header badge
        badge_text = f"[{action_label}: {det.entity_type}]"
        badge_top = max(0, box_coords[1] - 14)
        badge_bottom = box_coords[1]
        badge_right = min(base_img.width, box_coords[0] + len(badge_text) * 7 + 6)

        draw_base.rectangle([box_coords[0], badge_top, badge_right, badge_bottom], fill=style["badge"])
        if font:
            draw_base.text((box_coords[0] + 3, badge_top + 1), badge_text, fill=(255, 255, 255, 255), font=font)

    # Composite overlay onto base image
    annotated = Image.alpha_composite(base_img, overlay)

    out_buf = io.BytesIO()
    if export_format.upper() in ("JPG", "JPEG"):
        annotated.convert("RGB").save(out_buf, format="JPEG", quality=95)
    else:
        annotated.save(out_buf, format="PNG")

    logger.info("Image visualization complete: highlighted %d detected regions.", highlight_count)
    return out_buf.getvalue(), highlight_count
