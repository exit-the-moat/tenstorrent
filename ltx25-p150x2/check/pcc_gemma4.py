"""ltx1: Gemma 4 (LTX-2.5) device encoder vs the CPU reference: token ids, every hidden state (49), final embeds."""
import json, os, time
import torch, ttnn
from loguru import logger
from tests.scripts.common import get_updated_device_params
from models.tt_dit.encoders.gemma.encoder_pair import GemmaTokenizerEncoderPair
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

REF = os.environ.get("REF", "/work/ref/r0")
TE = "/models/ltx25/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
DIT = "/models/ltx25/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
meta = json.load(open(f"{REF}/meta.json"))

def pcc(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-30))

params = get_updated_device_params({"trace_region_size": 0, "l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D})
fc = params.pop("fabric_config"); params.pop("fabric_tensix_config", None)
ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED,
                       ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), **params)
ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear)
pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=2, mesh_axis=1))
pair = GemmaTokenizerEncoderPair(TE, mesh_device=mesh, ccl_manager=ccl, parallel_config=pc, checkpoint_name=DIT,
                                 mode="av", dynamic_load=False)
t = time.time(); pair.ensure_loaded(); logger.info(f"loaded in {time.time()-t:.0f}s")
ref = torch.load(f"{REF}/gemma_hidden.pt", weights_only=False)
tok = pair.tokenizer(meta["prompt"], max_length=1024)
logger.info(f"mask equal to reference: {torch.equal(tok.attention_mask, ref['mask'].long())} real tokens {int(tok.attention_mask.sum())}")
tt_ids = ttnn.from_torch(tok.input_ids, device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
mask = pair.gemma_encoder.build_attn_mask(tok.attention_mask, 1024)
t = time.time()
hs = pair.gemma_encoder(tt_ids, tt_attn_mask=mask)
host = [ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float().reshape(1, 1024, -1) for h in hs]
logger.info(f"encoder forward {time.time()-t:.1f}s, {len(host)} states")
m = ref["mask"][0].bool()
R = ref["hidden"].float()
res = []
for i in range(49):
    res.append(round(pcc(host[i][0][m], R[0, :, :, i][m]), 5))
logger.info(f"PCC per hidden state (real tokens) idx0..47 = embed, L0..L46; idx48 ref: {res}")
logger.info(f"ref idx48 vs device final_norm {pcc(host[49][0][m], R[0,:,:,48][m]):.5f}, vs device L47 raw {pcc(host[48][0][m], R[0,:,:,48][m]):.5f}")
t = time.time()
enc = pair.encode([meta["prompt"]])
c = torch.load(f"{REF}/s1_context.pt", weights_only=False)
v, a = enc[0]
logger.info(f"full encode {time.time()-t:.1f}s; final video embeds PCC {pcc(v, c['v_context'].float()):.5f} audio {pcc(a, c['a_context'].float()):.5f} shapes {tuple(v.shape)} {tuple(a.shape)}")
with open("/out/pcc_gemma.jsonl", "a") as f:
    f.write(json.dumps({"ref": REF, "hidden": res, "video": pcc(v, c['v_context'].float()), "audio": pcc(a, c['a_context'].float())}) + "\n")
ttnn.close_mesh_device(mesh); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("PCC_DONE")
