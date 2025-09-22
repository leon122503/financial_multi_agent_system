from __future__ import annotations

import base64
import json
import os
import time
import hashlib
from typing import Any, Dict, List
from pathlib import Path

from openai import OpenAI

from src.langgraph_state import LangGraphMarketState
from src.config import OPENAI_API_KEY, OPENAI_MODEL

DEFAULT_MODEL = OPENAI_MODEL or "gpt-4o-mini"


def _safe_serialize(obj: Any) -> str:
    """Safely serialize any object to JSON string."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({"value": str(obj)})


def _get_image_hash(image_bytes: bytes) -> str:
    """Get hash of image bytes to detect duplicates."""
    return hashlib.md5(image_bytes).hexdigest()


def _is_duplicate_image(image_bytes: bytes, existing_hashes: set) -> bool:
    """Check if image is a duplicate based on content hash."""
    image_hash = _get_image_hash(image_bytes)
    return image_hash in existing_hashes


def code_interpreter_graphs(state: LangGraphMarketState) -> LangGraphMarketState:
    """
    Use OpenAI code interpreter to generate graphs from fundamentals
    based on the user's prompt.

    Inputs (from state):
      - prompt: str (user's charting/analysis prompt)
      - fundamentals: Dict[str, Any] (numerical/time series data, e.g., key metrics)

    Writes to state:
      - graph_summary: str (assistant's textual explanation)
      - graph_images: List[str] (local file paths or file refs)
      - code_interpreter_raw: str (truncated raw output for debugging)
      - code_interpreter_error: str (if something fails)
    """
    prompt: str = (state.get("prompt") or "").strip()
    fundamentals: Dict[str, Any] = state.get("fundamentals") or {}

    if not OPENAI_API_KEY:
        return {"code_interpreter_error": "OPENAI_API_KEY not set"}

    if not prompt:
        return {"code_interpreter_error": "No prompt provided for graph generation"}

    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "You are a data science assistant using the code interpreter tool. "
        "You will receive 'fundamentals' as JSON and a user prompt describing what to visualize. "
        "Use Python with pandas/matplotlib/seaborn to create clear plots. "
        "Prefer sensible defaults (labels, titles, readable fonts). "
        "If the data is time series, plot vs time; otherwise choose bar/line/scatter appropriately. "
        "Display the plot and provide a concise textual summary of what the chart shows. "
        "Do not ask follow-up questions—make reasonable assumptions. "
        "\n"
        "You can create multiple different charts if they show different aspects of the data "
        "(e.g., one chart for revenue trends, another for profitability metrics, etc.). "
        "However, avoid creating duplicate charts that show the exact same information. "
        "Each chart should provide unique insights or perspectives on the financial data. "
        "\n"
        "Fundamentals shape hints: A top-level dict keyed by ticker (e.g., 'LULU'). "
        "Each value may include scalar fields (market_cap, margins), a 'valuation' dict, a 'profitability' dict, "
        "and arrays like 'annual_eps' or 'quarterly_eps' with items {date, eps}. "
        "When present, convert date strings to pandas datetime, coerce numerics with pd.to_numeric(errors='coerce'), "
        "drop None/NaN rows as needed, and build DataFrames for plotting. "
        "If multiple tickers, align series by date and include a legend. "
        "Always show() the plot; savefig if helpful. "
        "Create meaningful, distinct visualizations that complement each other."
    )

    user = (
        "Fundamentals (JSON):\n"
        + _safe_serialize(fundamentals)
        + "\n\n"
        + "User prompt for charts:\n"
        + prompt
    )

    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        # let tool run
        time.sleep(2)
    except Exception as e:
        return {"code_interpreter_error": f"OpenAI request failed: {e}"}

    summary_text = getattr(resp, "output_text", "").strip()

    # Extract images from container files
    saved_images: List[str] = []
    image_hashes: set = set()  # Track image hashes to prevent duplicates

    if hasattr(resp, "output") and resp.output:
        OUTPUT_DIR = Path("graph/output")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idx = 1

        for item in resp.output:
            # Check for message items with file citations
            if hasattr(item, "type") and item.type == "message":
                if hasattr(item, "content") and item.content:
                    for content_item in item.content:
                        if (
                            hasattr(content_item, "type")
                            and content_item.type == "output_text"
                        ):
                            if (
                                hasattr(content_item, "annotations")
                                and content_item.annotations
                            ):
                                for annotation in content_item.annotations:
                                    # Check for container file citations
                                    if (
                                        hasattr(annotation, "type")
                                        and annotation.type == "container_file_citation"
                                        and hasattr(annotation, "file_id")
                                        and hasattr(annotation, "container_id")
                                    ):

                                        file_id = annotation.file_id
                                        container_id = annotation.container_id
                                        filename = getattr(
                                            annotation, "filename", f"graph_{idx}.png"
                                        )

                                        try:
                                            # Download the container file using the correct API
                                            file_content = client.containers.files.content.retrieve(
                                                file_id, container_id=container_id
                                            )

                                            # The result is an HttpxBinaryResponseContent object with binary data
                                            content_bytes = file_content.content

                                            # Check for duplicates
                                            if _is_duplicate_image(
                                                content_bytes, image_hashes
                                            ):
                                                print(
                                                    f"   🔄 Skipping duplicate image: {filename}"
                                                )
                                                continue

                                            # Add hash to set and save file
                                            image_hashes.add(
                                                _get_image_hash(content_bytes)
                                            )
                                            path = (
                                                OUTPUT_DIR / f"graph_{idx}_{filename}"
                                            )
                                            with open(path, "wb") as f:
                                                f.write(content_bytes)

                                            saved_images.append(str(path))
                                            print(f"   💾 Saved unique chart: {path}")
                                            idx += 1

                                        except Exception as e:
                                            saved_images.append(
                                                f"failed_to_download:{file_id}:{e}"
                                            )

            # Also check for code interpreter calls that might have outputs with images
            elif hasattr(item, "type") and item.type == "code_interpreter_call":
                if hasattr(item, "outputs") and item.outputs:
                    for output_item in item.outputs:
                        if hasattr(output_item, "type") and output_item.type == "image":
                            # Handle inline base64 images from code interpreter outputs
                            if hasattr(output_item, "image") and hasattr(
                                output_item.image, "base64"
                            ):
                                try:
                                    img_bytes = base64.b64decode(
                                        output_item.image.base64
                                    )

                                    # Check for duplicates
                                    if _is_duplicate_image(img_bytes, image_hashes):
                                        print(f"   🔄 Skipping duplicate base64 image")
                                        continue

                                    # Add hash to set and save file
                                    image_hashes.add(_get_image_hash(img_bytes))
                                    path = OUTPUT_DIR / f"graph_{idx}.png"
                                    with open(path, "wb") as f:
                                        f.write(img_bytes)
                                    saved_images.append(str(path))
                                    print(f"   💾 Saved unique chart: {path}")
                                    idx += 1
                                except Exception as e:
                                    saved_images.append(f"failed_to_decode_base64:{e}")

    result: LangGraphMarketState = {
        "graph_summary": summary_text[:4000],
        "graph_images": saved_images,
        "code_interpreter_raw": str(resp)[:4000],
    }

    if not summary_text:
        result["code_interpreter_error"] = "No response text from code interpreter"

    # Log final results
    print(
        f"   📊 Generated {len(saved_images)} unique chart(s) (duplicates filtered out)"
    )
    for img_path in saved_images:
        print(f"     • {img_path}")

    return result
