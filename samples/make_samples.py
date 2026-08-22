"""테스트/데모용 샘플 PNG를 생성한다. 실행: uv run python samples/make_samples.py

외부 의존 없이 PNG 청크를 직접 만든다(프롬프트 메타데이터 포함).
"""

import json
import struct
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def chunk(ctype: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + ctype + data
        + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
    )


def make_png(text_chunks: dict[str, str], width: int = 8, height: int = 8) -> bytes:
    """텍스트 청크를 포함한 최소한의 PNG 바이트를 만든다."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes([70, 110, 180]) * width for _ in range(height))
    out = PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw))
    for key, value in text_chunks.items():
        out += chunk(b"tEXt", key.encode("utf-8") + b"\x00" + value.encode("utf-8"))
    return out + chunk(b"IEND", b"")


A1111_TEXT = (
    "masterpiece, best quality, 1girl, silver hair, school uniform, detailed background\n"
    "Negative prompt: (worst quality:1.4), low quality, blurry, bad hands\n"
    "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 31337, Size: 832x1216, "
    "Model hash: abcdef0123, Model: animagineXL, Version: v1.10.0"
)

COMFY_GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
    "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat wearing sunglasses, cinematic lighting, 4k", "clip": ["1", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, watermark", "clip": ["1", 1]}},
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 123456789, "steps": 30, "cfg": 7.5, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0, "model": ["1", 0],
            "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["4", 0],
        },
    },
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
}

COMFY_WORKFLOW = {
    "nodes": [
        {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["sd_xl_base_1.0.safetensors"]},
        {"id": 6, "type": "CLIPTextEncode", "widgets_values": ["a cat wearing sunglasses"], "inputs": []},
        {"id": 7, "type": "CLIPTextEncode", "widgets_values": ["blurry, low quality"], "inputs": []},
        {
            "id": 3, "type": "KSampler",
            "widgets_values": [123, "randomize", 30, 7.5, "euler", "normal", 1.0],
            "inputs": [
                {"name": "positive", "type": "CONDITIONING", "link": 1},
                {"name": "negative", "type": "CONDITIONING", "link": 2},
            ],
        },
    ],
    "links": [
        [1, 6, 0, 3, 0, "CONDITIONING"],
        [2, 7, 0, 3, 1, "CONDITIONING"],
    ],
}

# CLIPTextEncode의 text가 문자열이 아닌 링크(LLM 프롬프트 생성기 출력)인 워크플로.
# 최종 프롬프트는 ShowText 노드의 text_0에 남은 결과값.
COMFY_LLM_GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 773810126793850, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["16", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["13", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["28", 0], "clip": ["18", 0]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, ugly, bad, bad hands, bad anatomy, low res,", "clip": ["18", 0]}},
    "13": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1536, "batch_size": 1}},
    "16": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
    "17": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "18": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
    "28": {"class_type": "OpenAI Prompt Generator", "inputs": {"model": "gpt-5.5", "prompt_context": "주어지는 문장 / 키워드를 활용하여 이미지 생성용 프롬프트를 완전한 영어 문장으로 작성하라.", "additional_instructions": ["41", 0], "max_tokens": 4096, "temperature": 1.0, "seed": 637649833713935}},
    "34": {"class_type": "ShowText|pysssss", "inputs": {"text_0": "A cute young Asian woman in her early twenties stands confidently beside a tall modern observation tower, wearing high-waisted black leggings and a cropped oversized hoodie. The scene takes place in an open urban plaza at golden hour.", "text": ["28", 0]}},
    "41": {"class_type": "DPRandomGenerator", "inputs": {"text": "a cute young Asian woman wearing __leggings__, __background__", "seed": 1620, "autorefresh": "No", "is_changed": [float("nan")] }},
}

NAI_COMMENT = json.dumps({
    "prompt": "1girl, silver hair",
    "steps": 28, "sampler": "k_euler_ancestral", "seed": 987654321, "scale": 5,
    "uc": "lowres, bad",
})


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    (out_dir / "sample_a1111.png").write_bytes(make_png({"parameters": A1111_TEXT}))
    (out_dir / "sample_comfyui.png").write_bytes(
        make_png({"prompt": json.dumps(COMFY_GRAPH), "workflow": json.dumps(COMFY_WORKFLOW)})
    )
    (out_dir / "sample_comfyui_workflow_only.png").write_bytes(
        make_png({"workflow": json.dumps(COMFY_WORKFLOW)})
    )
    (out_dir / "sample_comfyui_linked.png").write_bytes(
        make_png({"prompt": json.dumps(COMFY_LLM_GRAPH)})
    )
    (out_dir / "sample_novelai.png").write_bytes(
        make_png({
            "Title": "1girl, silver hair, novelai",
            "Description": "lowres, bad",
            "Software": "NovelAI",
            "Source": "Stable Diffusion XL C1E1DE52",
            "Comment": NAI_COMMENT,
        })
    )
    (out_dir / "sample_plain.png").write_bytes(make_png({}))
    print(f"샘플 PNG 생성 완료: {out_dir}")


if __name__ == "__main__":
    main()
