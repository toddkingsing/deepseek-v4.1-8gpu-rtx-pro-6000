"""Bounded SM120 grouped GEMV for wo_a (configurable T threshold 1..64) + load-time cached BF16 fallback."""
from pathlib import Path
import ast, hashlib

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/sm120_gemv_bmm.py")
raw = p.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "8011bf16ecd5a9a3676fb9b870ec068f8189da0a3942a8e75b8236c0a00091c7"
source = raw.decode()
backup = p.with_name("speed9_bmm_reference.py")
backup.write_bytes(raw)

needle = "        self._mod._ext()\n        layer.sm120_gemv_bmm = self"
replacement = '''        self._mod._ext()
        self._langchao_max_t = int(os.environ.get("VLLM_LANGCHAO_BMM_MAX_T", "16"))
        if not (0 <= self._langchao_max_t <= self._mod.MAX_GROUPED_GEMV_T):
            raise ValueError(f"Unsupported VLLM_LANGCHAO_BMM_MAX_T={self._langchao_max_t}")
        layer.register_buffer(
            "_langchao_bmm_bf16",
            dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale).contiguous(),
            persistent=False,
        )
        logger.info_once(
            "langchao speed9 bounded wo_a: FP8 GEMV for T <= %d, cached BF16 BMM above",
            self._langchao_max_t,
        )
        layer.sm120_gemv_bmm = self'''
assert source.count(needle) == 1
source = source.replace(needle, replacement)

needle2 = "if tokens <= self._mod.MAX_GROUPED_GEMV_T:"
assert source.count(needle2) == 1
source = source.replace(needle2, "if tokens <= self._langchao_max_t:")

needle3 = "            w = dequant_mxfp8_to_bf16(wo_a.weight, wo_a.weight_scale)"
assert source.count(needle3) == 1
source = source.replace(needle3, "            w = wo_a._langchao_bmm_bf16")

ast.parse(source)
p.write_text(source)
print("Patched bounded sm120_gemv_bmm.py cleanly")
