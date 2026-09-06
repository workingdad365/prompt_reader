"""PNG 텍스트 청크를 파싱해 Stable Diffusion 계열 이미지의 생성 프롬프트를 추출한다.

지원 생성기:
- Automatic1111 / Stable Diffusion WebUI 및 포크(Forge 등) — ``parameters`` 청크
- ComfyUI — ``prompt``(API 그래프) / ``workflow`` 청크
- NovelAI — ``Title`` / ``Description`` / ``Comment`` 청크
- InvokeAI 구버전 — ``sd-metadata`` 청크
"""

from __future__ import annotations

import json
import re
import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_CHUNKS = 100_000
MAX_TEXT_BYTES = 32 * 1024 * 1024


class NotPngError(ValueError):
    """PNG 시그니처가 없는 파일."""


def _decode_text_value(raw: bytes) -> str:
    # tEXt/zTXt는 인코딩이 규격으로 고정되어 있지 않아 UTF-8 우선, 실패 시 latin-1.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", "replace")


def read_png_info(data: bytes) -> tuple[int, int, dict[str, str]]:
    """PNG 바이트에서 (가로, 세로, {텍스트 청크 키: 값})을 반환한다.

    tEXt / zTXt / iTXt 청크를 모두 읽는다. 잘렸거나 손상된 파일은 예외 없이
    여기까지 읽은 내용만 돌려준다.
    """
    if not data.startswith(PNG_SIGNATURE):
        raise NotPngError("PNG 시그니처가 없습니다")

    width = height = 0
    texts: dict[str, str] = {}
    pos, size = 8, len(data)
    for _ in range(MAX_CHUNKS):
        if pos + 8 > size:
            break
        length = int.from_bytes(data[pos:pos + 4], "big")
        ctype = data[pos + 4:pos + 8]
        start, end = pos + 8, pos + 8 + length
        if length > 0x7FFF_FFFF or end + 4 > size:
            break
        chunk = data[start:end]
        if ctype == b"IHDR" and length >= 8:
            width, height = struct.unpack(">II", chunk[:8])
        elif ctype == b"tEXt":
            key, _, value = chunk.partition(b"\x00")
            if key:
                texts[key.decode("latin-1", "replace")] = _decode_text_value(value)
        elif ctype == b"zTXt":
            key, _, rest = chunk.partition(b"\x00")
            if key and rest:
                try:
                    value = zlib.decompressobj().decompress(rest[1:], MAX_TEXT_BYTES)
                except zlib.error:
                    value = b""
                if value:
                    texts[key.decode("latin-1", "replace")] = _decode_text_value(value)
        elif ctype == b"iTXt":
            key, _, rest = chunk.partition(b"\x00")
            if key and len(rest) >= 2:
                compressed = rest[0] == 1
                body = rest[2:]
                _, _, body = body.partition(b"\x00")  # language tag
                _, _, body = body.partition(b"\x00")  # translated keyword
                if compressed:
                    try:
                        body = zlib.decompressobj().decompress(body, MAX_TEXT_BYTES)
                    except zlib.error:
                        body = b""
                texts[key.decode("latin-1", "replace")] = body.decode("utf-8", "replace")
        elif ctype == b"IEND":
            break
        pos = end + 4
    return width, height, texts


def _parse_a1111(text: str) -> dict:
    """``parameters`` 청크(A1111 계열)를 positive/negative/settings로 나눈다."""
    lines = text.splitlines()
    neg_idx = None
    set_idx = None
    for i, line in enumerate(lines):
        if neg_idx is None and line.startswith("Negative prompt:"):
            neg_idx = i
        elif set_idx is None and (
            re.match(r"^Steps:\s*\d", line) or line.startswith("Denoising strength:")
        ):
            set_idx = i
    pos_end = neg_idx if neg_idx is not None else set_idx
    positive = "\n".join(lines[:pos_end if pos_end is not None else len(lines)]).strip() or None

    negative = None
    if neg_idx is not None:
        end = set_idx if set_idx is not None and set_idx > neg_idx else len(lines)
        neg_lines = lines[neg_idx:end]
        neg_lines[0] = neg_lines[0][len("Negative prompt:"):].strip()
        negative = "\n".join(neg_lines).strip() or None

    settings = "\n".join(lines[set_idx:]).strip() or None if set_idx is not None else None
    return {"positive": positive, "negative": negative, "settings": settings}


# --- ComfyUI -----------------------------------------------------------------

_TEXT_INPUT_KEYS = ("text", "text_g", "text_l")
# 링크를 거슬러 올라갈 때 값으로 인정할 정적 텍스트 위젯 키
_STATIC_TEXT_KEYS = ("text", "text_g", "text_l", "string", "value", "prompt")
# 실행 결과를 자기 위젯에 되써 넣는(=직전 실행 값이 남는) 미리보기 노드
_ECHO_NODE_HINTS = ("showtext", "showanything", "preview", "display")
# ConditioningCombine / ControlNetApply 처럼 positive/negative를 그대로 흘려보내는 노드의 입력 키
_PASSTHROUGH_KEYS = (
    "positive",
    "negative",
    "conditioning_1",
    "conditioning_2",
    "conditioning_to",
    "conditioning_from",
)


def _encoded_texts(node: dict, nodes: dict) -> list[str]:
    """CLIPTextEncode 계열 노드에서 프롬프트 문자열을 꺼낸다.

    ``text`` 입력이 링크(예: LLM 프롬프트 생성기 → CLIPTextEncode)면 연결된
    노드의 결과값을 그래프에서 찾아 해석한다.
    """
    if not str(node.get("class_type", "")).startswith("CLIPTextEncode"):
        return []
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return []
    found = []
    for key in _TEXT_INPUT_KEYS:
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
        elif isinstance(value, (list, tuple)) and value:
            resolved = _resolve_linked_text(value, nodes)
            if resolved:
                found.append(resolved)
    return found


def _resolve_linked_text(ref, nodes: dict, depth: int = 0) -> str | None:
    """링크(["노드ID", 슬롯])를 거슬러 올라가 정적 텍스트 위젯 값을 찾는다.

    ComfyUI API 그래프는 노드의 '출력'을 저장하지 않는다. ShowText 계열에 남은
    ``text_0``은 큐 제출 시점의 위젯 값, 즉 **직전 실행의 결과**다. LLM이나
    와일드카드처럼 매 실행마다 결과가 바뀌는 노드에서는 실제 사용된 프롬프트와
    달라지므로 사용하지 않는다.
    """
    if depth > 5 or not isinstance(ref, (list, tuple)) or not ref:
        return None
    node = nodes.get(str(ref[0]))
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None

    class_type = str(node.get("class_type", "")).lower()
    is_echo = any(hint in class_type for hint in _ECHO_NODE_HINTS)

    for key in _STATIC_TEXT_KEYS:
        value = inputs.get(key)
        if isinstance(value, str) and value.strip() and not is_echo:
            return value.strip()
        if isinstance(value, (list, tuple)) and value:
            resolved = _resolve_linked_text(value, nodes, depth + 1)
            if resolved:
                return resolved
    return None


def _follow_conditioning(ref, nodes: dict, depth: int = 0) -> list[str]:
    """샘플러의 positive/negative 입력(링크)을 따라가 텍스트 인코더까지 도달한다."""
    if depth > 5 or not isinstance(ref, (list, tuple)) or not ref:
        return []
    node = nodes.get(str(ref[0]))
    if not isinstance(node, dict):
        return []
    texts = _encoded_texts(node, nodes)
    if texts:
        return texts
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return []
    collected: list[str] = []
    for key in _PASSTHROUGH_KEYS:
        collected.extend(_follow_conditioning(inputs.get(key), nodes, depth + 1))
    return collected


def _comfy_settings(nodes: dict, samplers: list[dict]) -> str | None:
    parts: list[str] = []
    for node in nodes.values():
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        name = None
        if class_type in ("CheckpointLoaderSimple", "CheckpointLoader"):
            name = inputs.get("ckpt_name")
        elif class_type in ("UNETLoader", "UnetLoaderGGUF"):
            name = inputs.get("unet_name")
        if isinstance(name, str) and name.strip():
            parts.append(f"Model: {name.strip()}")
            break
    for loader_type, key, label in (
        ("CLIPLoader", "clip_name", "CLIP"),
        ("VAELoader", "vae_name", "VAE"),
    ):
        for node in nodes.values():
            if str(node.get("class_type", "")) == loader_type:
                name = node.get("inputs", {}).get(key)
                if isinstance(name, str) and name.strip():
                    parts.append(f"{label}: {name.strip()}")
                    break
    if samplers:
        inputs = samplers[0].get("inputs", {})
        for key, label in (
            ("steps", "Steps"),
            ("cfg", "CFG"),
            ("sampler_name", "Sampler"),
            ("scheduler", "Scheduler"),
            ("denoise", "Denoise"),
        ):
            value = inputs.get(key)
            if isinstance(value, (int, float, str)):
                parts.append(f"{label}: {value}")
        seed = inputs.get("seed")
        if not isinstance(seed, (int, float)):
            seed = inputs.get("noise_seed")
        if isinstance(seed, (int, float)):
            parts.append(f"Seed: {seed}")
    return ", ".join(parts) if parts else None


def _parse_comfyui_api(graph: dict) -> dict:
    """ComfyUI의 API 그래프(``prompt`` 청크)에서 프롬프트를 복원한다."""
    nodes = {
        str(k): v for k, v in graph.items() if isinstance(v, dict) and isinstance(v.get("inputs"), dict)
    }
    samplers = [n for n in nodes.values() if "positive" in n["inputs"] and "negative" in n["inputs"]]

    positives: list[str] = []
    negatives: list[str] = []
    for sampler in samplers:
        positives.extend(_follow_conditioning(sampler["inputs"].get("positive"), nodes))
        negatives.extend(_follow_conditioning(sampler["inputs"].get("negative"), nodes))

    if not positives and not negatives:
        # 샘플러를 추적할 수 없는 그래프면 모든 텍스트 인코더를 나열
        for node in nodes.values():
            positives.extend(_encoded_texts(node, nodes))

    return {
        "positive": "\n\n".join(_dedupe(positives)) or None,
        "negative": "\n\n".join(_dedupe(negatives)) or None,
        "settings": _comfy_settings(nodes, samplers),
    }


def _parse_comfyui_workflow(wf: dict) -> dict:
    """API 그래프(prompt 청크)가 없을 때 UI workflow 청크로 최대한 복원한다."""
    nodes = [n for n in wf.get("nodes", []) if isinstance(n, dict)]
    by_id = {str(n.get("id")): n for n in nodes}
    links = {str(l[0]): l for l in wf.get("links", []) if isinstance(l, (list, tuple)) and len(l) >= 4}

    def encoded_texts(node) -> list[str]:
        if not isinstance(node, dict) or not str(node.get("type", "")).startswith("CLIPTextEncode"):
            return []
        return [v.strip() for v in node.get("widgets_values", []) if isinstance(v, str) and v.strip()]

    positives: list[str] = []
    negatives: list[str] = []
    for node in nodes:
        inputs = node.get("inputs")
        if not isinstance(inputs, list):
            continue
        by_name = {i.get("name"): i.get("link") for i in inputs if isinstance(i, dict)}
        if "positive" not in by_name or "negative" not in by_name:
            continue
        for field, bucket in (("positive", positives), ("negative", negatives)):
            link = links.get(str(by_name.get(field)))
            if link is not None:
                bucket.extend(encoded_texts(by_id.get(str(link[1]))))

    if not positives and not negatives:
        for node in nodes:
            positives.extend(encoded_texts(node))

    return {
        "positive": "\n\n".join(_dedupe(positives)) or None,
        "negative": "\n\n".join(_dedupe(negatives)) or None,
        "settings": None,
    }


# --- NovelAI / InvokeAI ------------------------------------------------------

def _parse_novelai(texts: dict[str, str]) -> dict | None:
    comment_raw = texts.get("Comment")
    if not comment_raw:
        return None
    try:
        comment = json.loads(comment_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(comment, dict):
        return None
    looks_nai = texts.get("Software") == "NovelAI" or ("steps" in comment and "sampler" in comment)
    if not looks_nai:
        return None
    positive = (texts.get("Title") or "").strip()
    if not positive:
        return None
    negative = (texts.get("Description") or "").strip()
    if not negative and isinstance(comment.get("uc"), str):
        negative = comment["uc"].strip()
    parts = [
        f"{key}: {comment[key]}"
        for key in ("steps", "sampler", "seed", "scale", "strength", "noise_schedule")
        if key in comment
    ]
    return {"positive": positive, "negative": negative or None, "settings": ", ".join(parts) or None}


def _parse_invokeai(texts: dict[str, str]) -> dict | None:
    raw = texts.get("sd-metadata")
    if not raw:
        return None
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(meta, dict):
        return None
    image = meta.get("image") if isinstance(meta.get("image"), dict) else meta
    positive = image.get("prompt")
    if not isinstance(positive, str) or not positive.strip():
        return None
    negative = image.get("negative_prompt")
    if isinstance(negative, list):
        negative = ", ".join(str(n) for n in negative)
    negative = negative.strip() if isinstance(negative, str) else None
    parts = [f"{k}: {image[k]}" for k in ("steps", "sampler", "seed", "cfg_scale") if k in image]
    return {
        "positive": positive.strip(),
        "negative": negative or None,
        "settings": ", ".join(parts) or None,
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def pretty_json(text: str) -> str | None:
    """텍스트 청크가 JSON이면 사람이 읽기 좋게 정렬한 문자열을 반환한다.

    ComfyUI는 Python json 확장 표기(NaN 등)를 그대로 기록하는 경우가 있어서
    브라우저의 JSON.parse는 실패하고 Python은 성공한다. 그래서 정렬은 서버에서:
    ensure_ascii=False로 \\uXXXX 이스케이프를 실제 한국어 등으로 풀어낸다.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return json.dumps(parsed, indent=2, ensure_ascii=False)


# 프롬프트 폴백 탐색에서 제외할, 이미 전용 파서가 있는 청크 키
_HANDLED_CHUNK_KEYS = {
    "prompt",
    "workflow",
    "parameters",
    "sd-metadata",
    "Title",
    "Description",
    "Comment",
    "Software",
    "Source",
    "Generation time",
}


def _chunk_prompt_fallback(texts: dict[str, str]) -> tuple[str | None, str | None]:
    """AddMetaData(Mikey) 같은 노드가 남긴 커스텀 청크에서 프롬프트를 찾는다.

    ComfyUI 그래프로는 복원할 수 없는 동적 프롬프트(LLM 출력, 와일드카드 전개)를
    이런 청크가 실행 시점의 실제 값으로 담고 있다.
    """
    for key, value in texts.items():
        lowered = key.lower()
        if key in _HANDLED_CHUNK_KEYS or "prompt" not in lowered:
            continue
        if "negative" in lowered:
            continue
        text = value.strip()
        # Mikey의 AddMetaData는 값을 JSON 문자열로 감싸 기록한다
        if text.startswith('"') and text.endswith('"'):
            try:
                decoded = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                decoded = None
            if isinstance(decoded, str):
                text = decoded.strip()
        if text:
            return text, key
    return None, None


def extract_prompt_info(png_bytes: bytes) -> dict:
    """PNG 바이트를 받아 프롬프트 정보 구조를 반환한다."""
    width, height, texts = read_png_info(png_bytes)
    result = {
        "width": width,
        "height": height,
        "source": None,
        "found": False,
        "positive": None,
        "positive_source": None,
        "negative": None,
        "settings": None,
        "chunks": texts,
    }

    parsed: dict | None = None

    if "parameters" in texts:
        parsed = _parse_a1111(texts["parameters"])
        result["source"] = "Automatic1111 / SD WebUI"
    elif "prompt" in texts:
        try:
            graph = json.loads(texts["prompt"])
        except json.JSONDecodeError:
            graph = None
        if isinstance(graph, dict) and any(
            isinstance(v, dict) and "class_type" in v for v in graph.values()
        ):
            parsed = _parse_comfyui_api(graph)
            result["source"] = "ComfyUI"

    if parsed is None and "workflow" in texts:
        try:
            wf = json.loads(texts["workflow"])
        except json.JSONDecodeError:
            wf = None
        if isinstance(wf, dict) and isinstance(wf.get("nodes"), list):
            parsed = _parse_comfyui_workflow(wf)
            result["source"] = "ComfyUI (workflow)"

    if parsed is None:
        parsed = _parse_novelai(texts)
        if parsed is not None:
            result["source"] = "NovelAI"

    if parsed is None:
        parsed = _parse_invokeai(texts)
        if parsed is not None:
            result["source"] = "InvokeAI"

    if parsed is not None:
        result["positive"] = parsed.get("positive")
        result["negative"] = parsed.get("negative")
        result["settings"] = parsed.get("settings")

    if not result["positive"]:
        text, key = _chunk_prompt_fallback(texts)
        if text:
            result["positive"] = text
            result["positive_source"] = key

    result["found"] = bool(result["positive"] or result["negative"])
    return result
