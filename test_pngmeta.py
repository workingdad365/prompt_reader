"""pngmeta 모듈 테스트. 실행: uv run python test_pngmeta.py"""

import json
import struct
import zlib

from pngmeta import NotPngError, extract_prompt_info, pretty_json, read_png_info
from samples.make_samples import COMFY_GRAPH, COMFY_LLM_GRAPH, COMFY_WORKFLOW, chunk, make_png


def test_a1111_basic():
    info = extract_prompt_info(make_png({"parameters": (
        "masterpiece, 1girl\nsecond line of prompt\n"
        "Negative prompt: bad, worse\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512"
    )}))
    assert info["source"] == "Automatic1111 / SD WebUI"
    assert info["positive"] == "masterpiece, 1girl\nsecond line of prompt"
    assert info["negative"] == "bad, worse"
    assert info["settings"] == "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512"
    assert info["found"] is True
    assert (info["width"], info["height"]) == (8, 8)


def test_a1111_multiline_negative():
    info = extract_prompt_info(make_png({"parameters": (
        "a portrait\n"
        "Negative prompt: line one of negative\ncontinues on second line\n"
        "Steps: 10, Seed: 5"
    )}))
    assert info["positive"] == "a portrait"
    assert info["negative"] == "line one of negative\ncontinues on second line"
    assert info["settings"] == "Steps: 10, Seed: 5"


def test_a1111_no_negative():
    info = extract_prompt_info(make_png({"parameters": "just a prompt\nSteps: 10, Seed: 2"}))
    assert info["negative"] is None
    assert info["positive"] == "just a prompt"
    assert info["found"] is True


def test_comfyui_api():
    info = extract_prompt_info(make_png({
        "prompt": json.dumps(COMFY_GRAPH),
        "workflow": "not json (무시되어야 함)",
    }))
    assert info["source"] == "ComfyUI"
    assert "cat wearing sunglasses" in info["positive"]
    assert "blurry, low quality, watermark" in info["negative"]
    assert "Steps: 30" in info["settings"]
    assert "Seed: 123456789" in info["settings"]
    assert "sd_xl_base_1.0.safetensors" in info["settings"]
    assert info["found"] is True


def test_comfyui_workflow_only():
    info = extract_prompt_info(make_png({"workflow": json.dumps(COMFY_WORKFLOW)}))
    assert info["source"] == "ComfyUI (workflow)"
    assert "cat wearing sunglasses" in info["positive"]
    assert "blurry, low quality" in info["negative"]


def test_comfyui_sdxl_dual_text():
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
        "9": {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text_g": "four-legged cat", "text_l": "a cat", "clip": ["1", 1]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "negation", "clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["9", 0], "negative": ["10", 0]}},
    }
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert "four-legged cat" in info["positive"]
    assert "a cat" in info["positive"]
    assert info["negative"] == "negation"


def test_comfyui_conditioning_chain():
    graph = {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "base pos"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "base neg"}},
        "20": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["6", 0], "conditioning_2": ["6", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["20", 0], "negative": ["7", 0]}},
    }
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] == "base pos"
    assert info["negative"] == "base neg"


def test_comfyui_no_sampler_fallback():
    graph = {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "orphan text"}},
    }
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] == "orphan text"


def test_comfyui_linked_prompt_generator():
    """ShowText의 text_0은 직전 실행의 값이므로 프롬프트로 쓰지 않는다."""
    info = extract_prompt_info(make_png({"prompt": json.dumps(COMFY_LLM_GRAPH)}))
    assert info["positive"] is None
    assert info["negative"].startswith("blurry, ugly, bad")
    assert "z_image_turbo_bf16.safetensors" in info["settings"]  # UNETLoader → Model
    assert "qwen_3_4b.safetensors" in info["settings"]  # CLIPLoader → CLIP
    assert "ae.safetensors" in info["settings"]  # VAELoader → VAE
    assert "Steps: 8" in info["settings"]
    assert "Seed: 773810126793850" in info["settings"]


def test_comfyui_custom_chunk_prompt():
    """AddMetaData 등이 남긴 커스텀 청크는 실행 시점 값이므로 프롬프트로 쓴다."""
    info = extract_prompt_info(make_png({
        "prompt": json.dumps(COMFY_LLM_GRAPH),
        "Final Prompt": json.dumps("A woman standing among stone corridors"),
    }))
    assert info["positive"] == "A woman standing among stone corridors"
    assert info["positive_source"] == "Final Prompt"
    assert info["found"] is True


def test_comfyui_linked_text_fallback_source():
    """소비 노드에 결과값이 없으면 소스 노드 자체의 문자열 입력으로 폴백."""
    graph = {
        "5": {"class_type": "DPRandomGenerator", "inputs": {"text": "template with __wildcard__"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["5", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg"}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
    }
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] == "template with __wildcard__"


def qwen_graph(refine=False):
    return {
        "459:452": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "prompt": ["459:472", 0], "negative_prompt": "blur"}},
        "459:472": {"class_type": "PreviewAny", "inputs": {
            "source": ["459:474", 0], "text": "stale preview"}},
        "459:474": {"class_type": "ComfySwitchNode", "inputs": {
            "switch": refine, "on_false": "원본 portrait", "on_true": ["459:471", 0]}},
        "459:471": {"class_type": "TextGenerate", "inputs": {
            "prompt": "원본 portrait", "system_prompt": "not an image prompt"}},
        "459:458": {"class_type": "KSampler", "inputs": {
            "positive": ["459:452", 0], "negative": ["459:452", 1]}},
    }


def test_qwen_refine_off():
    info = extract_prompt_info(make_png({"prompt": json.dumps(qwen_graph())}))
    assert info["found"] is True
    assert info["positive"] == "원본 portrait"
    assert info["negative"] == "blur"
    assert info["prompt_warning"] is None


def test_qwen_refine_on_original_is_labeled():
    info = extract_prompt_info(make_png({"prompt": json.dumps(qwen_graph(True))}))
    assert info["found"] is True
    assert info["positive"] == "원본 portrait"
    assert info["negative"] == "blur"
    assert "보강 전" in info["prompt_warning"]


def test_qwen_final_chunk_overrides_original():
    info = extract_prompt_info(make_png({
        "prompt": json.dumps(qwen_graph(True)),
        "Final Prompt": json.dumps("actual enhanced output"),
    }))
    assert info["positive"] == "actual enhanced output"
    assert info["positive_source"] == "Final Prompt"
    assert info["prompt_warning"] is None


def test_qwen_direct_and_empty_positive():
    graph = qwen_graph()
    graph["459:452"]["inputs"]["prompt"] = "direct prompt"
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] == "direct prompt"
    assert info["negative"] == "blur"
    graph["459:452"]["inputs"]["prompt"] = ""
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] is None
    assert info["negative"] == "blur"


def test_qwen_unknown_switch_and_cycle():
    graph = qwen_graph()
    graph["459:474"]["inputs"]["switch"] = ["unknown", 0]
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] is None  # 선택되지 않은 분기를 추측하지 않는다
    graph["459:474"]["inputs"]["switch"] = False
    graph["459:474"]["inputs"]["on_false"] = ["459:472", 0]
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] is None


def test_qwen_no_sampler():
    graph = qwen_graph(True)
    del graph["459:458"]
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] == "원본 portrait"
    assert info["prompt_warning"]


def krea_graph(refine=True, style=False):
    graph = qwen_graph(refine)
    graph["459:452"] = {"class_type": "CLIPTextEncode", "inputs": {"text": ["style-switch", 0]}}
    graph["459:458"]["inputs"]["negative"] = ["zero", 0]
    graph["zero"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["459:452", 0]}}
    graph["refine"] = {"class_type": "PrimitiveBoolean", "inputs": {"value": refine}}
    graph["style"] = {"class_type": "PrimitiveBoolean", "inputs": {"value": style}}
    graph["459:474"]["inputs"]["switch"] = ["refine", 0]
    graph["459:471"]["inputs"]["prompt"] = ["instruction", 0]
    graph["instruction"] = {"class_type": "StringConcatenate", "inputs": {
        "string_a": ["prefix", 0], "string_b": "원본 portrait", "delimiter": ""}}
    graph["prefix"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "Expand this:\n\n"}}
    graph["style-switch"] = {"class_type": "ComfySwitchNode", "inputs": {
        "switch": ["style", 0], "on_false": ["459:472", 0], "on_true": ["suffix", 0]}}
    graph["suffix"] = {"class_type": "StringConcatenate", "inputs": {
        "string_a": ["459:472", 0], "string_b": "sketch style", "delimiter": ", "}}
    return graph


def test_krea_linked_switches_and_concatenation():
    for refine in (False, True):
        for style in (False, True):
            info = extract_prompt_info(make_png({"prompt": json.dumps(krea_graph(refine, style))}))
            expected = ("Expand this:\n\n" if refine else "") + "원본 portrait"
            expected += ", sketch style" if style else ""
            assert info["positive"] == expected
            assert bool(info["prompt_warning"]) == refine
            assert info["negative"] is None  # ZeroOut은 부정 프롬프트 텍스트가 아니다
            assert info["found"] is True


def test_krea_unresolved_and_empty_concatenation():
    graph = krea_graph(False, True)
    graph["suffix"]["inputs"]["string_b"] = ["missing", 0]
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] is None  # 일부만 읽고 완전한 프롬프트로 표시하지 않는다
    graph["suffix"]["inputs"]["string_b"] = ""
    graph["suffix"]["inputs"]["delimiter"] = ""
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] == "원본 portrait"
    graph["style"]["inputs"]["value"] = ["style", 0]
    info = extract_prompt_info(make_png({"prompt": json.dumps(graph)}))
    assert info["positive"] is None


def test_pretty_json_handles_nan_and_escapes():
    """Python JSON 확장(NaN)과 \\uXXXX 이스케이프를 정렬 시 해석해야 한다."""
    raw = '{"prompt_context": "\\uc8fc\\uc5b4\\uc9c0\\ub294 \\ubb38\\uc7a5", "is_changed": [NaN]}'
    pretty = pretty_json(raw)
    assert pretty is not None
    assert "주어지는 문장" in pretty
    assert "NaN" in pretty
    assert pretty_json("Steps: 20, Sampler: x") is None


def test_novelai():
    info = extract_prompt_info(make_png({
        "Title": "1girl, silver hair",
        "Description": "lowres, bad",
        "Software": "NovelAI",
        "Comment": json.dumps({"prompt": "1girl, silver hair", "steps": 28, "sampler": "k_euler", "seed": 42, "scale": 5, "uc": "lowres, bad"}),
    }))
    assert info["source"] == "NovelAI"
    assert info["positive"] == "1girl, silver hair"
    assert info["negative"] == "lowres, bad"
    assert "steps: 28" in info["settings"]


def test_ztxt_and_itxt_chunks():
    ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x10\x20\x30" * 4 for _ in range(4))
    ztxt = chunk(b"zTXt", b"parameters\x00\x00" + zlib.compress("hidden prompt\nSteps: 5".encode()))
    itxt = chunk(b"iTXt", b"prompt\x00\x00\x00ko\x00\x00" + b'{"n":1}')
    data = (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw))
        + ztxt + itxt + chunk(b"IEND", b"")
    )
    width, height, texts = read_png_info(data)
    assert (width, height) == (4, 4)
    assert texts["parameters"].startswith("hidden prompt")
    assert texts["prompt"] == '{"n":1}'


def test_korean_utf8_text():
    info = extract_prompt_info(make_png({"parameters": "한국어 프롬프트, 1girl\nSteps: 10"}))
    assert info["positive"] == "한국어 프롬프트, 1girl"


def test_no_metadata():
    info = extract_prompt_info(make_png({}))
    assert info["found"] is False
    assert info["source"] is None
    assert info["chunks"] == {}
    assert (info["width"], info["height"]) == (8, 8)


def test_not_png():
    for bad in (b"", b"GIF89a", b"\x89PNG\r\n\x1a"):
        try:
            read_png_info(bad)
            raise AssertionError(f"{bad!r}이 NotPngError를 발생시키지 않음")
        except NotPngError:
            pass


def test_truncated_png_no_crash():
    full = make_png({"parameters": "x\nSteps: 1"}, width=2, height=2)
    extract_prompt_info(full[: len(full) // 2])  # 예외 없이 실행되면 충분


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # 테스트 자체의 오류도 실패로 처리
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
