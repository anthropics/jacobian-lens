"""Interactive J-lens dashboard. Runs ON the vast.ai instance only.

Loads one quantized model + one published lens once, then serves a prompt box
that renders jlens' own slice visualisation. Bound to 127.0.0.1 by bootstrap.sh
and reached through an SSH tunnel, so there is no auth, no multi-user handling
and no persistence here by design.

    uvicorn dashboard:app --host 127.0.0.1 --port 7860
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments", "flood"))
from run import measure, wordlike_mask  # noqa: E402 - needs the sys.path line above

MODEL_ID = os.environ.get("JLENS_MODEL_ID", "google/gemma-4-12B-it")
LENS_REPO = os.environ.get("JLENS_LENS_REPO", "lamm-mit/gemma4-jacobian-lenses")
LENS_FILE = os.environ.get("JLENS_LENS_FILE", "gemma4-12b-it/paper/seed0.pt")
PRECISION = os.environ.get("JLENS_PRECISION", "bf16")
FLOOD_ITEMS = os.path.join(REPO, "experiments", "flood", "items.jsonl")
FLOOD_OUT = os.environ.get("JLENS_FLOOD_OUT", "/workspace/flood-results.jsonl")

# Hard caps. The page offers these as defaults; the server re-clamps because the
# form is trivially editable and a 512-token x stride-1 slice is a much bigger
# bill (two full-vocab argsort sweeps per row) than it looks in the UI.
MAX_SEQ_CAP = 512
TOP_N_CAP = 12
MAX_TRACKED = 256  # mandatory for mode="embed": the page inlines one rank array per tracked token
GEN_CAP = 128

EXAMPLES = [
    "Fact: The currency used in the country shaped like a boot is",
    "Fact: The capital city of the country that hosted the 2016 Summer Olympics is",
    "Fact: The element with atomic number 79, the one used in wedding rings, is called",
    "Fact: The author of the play in which the prince of Denmark sees his father's ghost was born in",
]

_lock = threading.Lock()  # jlens registers hooks on the shared blocks: one slice at a time
_state: dict = {"status": "loading", "detail": "starting", "error": None}
_model = _lens = _tokenizer = None

# The flood suite runs in-process so it reuses the loaded model, and takes _lock
# per item so /run can still interleave between items.
_flood: dict = {"running": False, "total": 0, "results": [], "error": None, "params": {}}
_flood_cv = threading.Condition()  # notified after every appended record and at the end
_flood_mask = None  # ~250k decodes; computed once, reused by every run


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load() -> None:
    """Load model + lens. Runs on a background thread so the page can serve a
    readable status (and a readable failure) during the ~15 min cold start."""
    global _model, _lens, _tokenizer
    try:
        import torch

        import jlens
        import jlens.vis as jvis

        if os.environ.get("JLENS_TINY") == "1":
            # CPU smoke test: the toy decoder from tests/, fitted the way run.py
            # --tiny does. Numbers mean nothing; the routes are exercised for real.
            sys.path.insert(0, os.path.join(REPO, "tests"))
            from tiny import TinyDecoder

            from jlens.fitting import fit

            model = TinyDecoder(n_layers=4, d_model=8)
            lens = fit(
                model,
                ["abcdefghij " * 5, "klmnopqrst " * 5],
                source_layers=[0, 1, 2],
                dim_batch=4,
                max_seq_len=64,
            )
            _tokenizer = model.tokenizer
            try:
                jvis._template("embed")
            except Exception as exc:  # noqa: BLE001 - non-fatal, retried per request
                log(f"WARNING: d3 prefetch failed ({exc}); /run retries before computing")
            _model, _lens = model, lens
            _state.update(
                status="ready",
                detail=f"TINY · {model.n_layers} layers · lens {len(lens.source_layers)} fitted layers",
            )
            log("ready (tiny)")
            return

        import transformers

        _state["detail"] = f"loading {MODEL_ID} ({PRECISION})"
        log(_state["detail"])
        if PRECISION == "nf4":
            quant = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["lm_head"],
            )
        elif PRECISION == "int8":
            quant = transformers.BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=["lm_head"]
            )
        elif PRECISION == "bf16":
            quant = None
        else:
            raise ValueError(f"JLENS_PRECISION must be nf4|int8|bf16, got {PRECISION!r}")

        t0 = time.time()
        # AutoModelForImageTextToText, not AutoModelForCausalLM: this checkpoint is a VLM
        # and the CausalLM alias resolves to a wrapper jlens cannot find the decoder in.
        # device_map={"": 0} not "auto": "auto" would scatter layers and reinstate a
        # per-call host->device copy of every Jacobian.
        hf_model = transformers.AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0}
        )
        _tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        model = jlens.from_hf(hf_model, _tokenizer, compile=False)
        log(f"model loaded in {time.time() - t0:.0f}s | {model!r} | layout={model.layout.path}")

        # unembed() casts residuals to the lm_head dtype. A quantized head would make
        # every readout silent garbage, so refuse to serve rather than mislead.
        head_dtype = model._lm_head.weight.dtype
        if head_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise RuntimeError(
                f"lm_head is {head_dtype}; it was quantized and every lens readout "
                "would be garbage. Check llm_int8_skip_modules."
            )

        _state["detail"] = f"loading lens {LENS_REPO}"
        log(_state["detail"])
        lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
        if lens.d_model != model.d_model:
            raise RuntimeError(f"lens d_model={lens.d_model} != model d_model={model.d_model}")
        bad = [n for n in lens.source_layers if not 0 <= n < model.n_layers]
        if bad:
            raise RuntimeError(f"lens layers {bad} out of range for {model.n_layers} layers")
        # Pin the Jacobians to the GPU once. transport() otherwise copies ~100MB per
        # layer per pass, twice per layer per request.
        lens.jacobians = {n: J.to(model.input_device) for n, J in lens.jacobians.items()}
        log(f"lens loaded | {lens!r}")

        _state["detail"] = "warming caches"
        log(_state["detail"])
        vocab = int(model._lm_head.weight.shape[0])
        # ~250k single-token decodes, cached for process lifetime. Paying it here keeps
        # it off the first request.
        jvis._meaningful_token_mask(_tokenizer, vocab, model.input_device)
        try:
            jvis._template("embed")  # one-off d3 fetch; embed mode needs it inlined
        except Exception as exc:  # noqa: BLE001 - non-fatal, retried per request
            log(f"WARNING: d3 prefetch failed ({exc}); /run retries before computing")

        peak = torch.cuda.max_memory_allocated() / 1024**3
        now = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"VRAM: {now:.1f} GiB resident, {peak:.1f} GiB peak, {total:.1f} GiB total")

        _model, _lens = model, lens
        _state.update(
            status="ready",
            detail=(
                f"{MODEL_ID} · {PRECISION} · {model.n_layers} layers · d_model {model.d_model} · "
                f"lens {len(lens.source_layers)} fitted layers · {peak:.1f}/{total:.0f} GiB VRAM peak"
            ),
        )
        log("ready")
    except Exception:
        _state.update(status="failed", detail="load failed", error=traceback.format_exc())
        log("LOAD FAILED\n" + _state["error"])


def _generate(context_token_ids: list[int], n_new: int, top_n: int) -> dict:
    """Greedy-decode ``n_new`` tokens from the exact ids the slice used, and
    return the continuation plus the top-``top_n`` next-token distribution at
    the last prompt position. This is what the L47 row is predicting, with
    probabilities, and unmasked (mask_display hides non-word tokens there)."""
    import torch

    hf = _model._hf_model
    ids = torch.tensor([context_token_ids], device=_model.input_device)
    out = hf.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=n_new,
        do_sample=False,
        pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
        output_logits=True,
        return_dict_in_generate=True,
    )
    new_ids = out.sequences[0, ids.shape[1] :].tolist()
    probs = torch.softmax(out.logits[0][0].float(), dim=-1)
    top = probs.topk(top_n)
    return {
        "continuation": _tokenizer.decode(new_ids, skip_special_tokens=False),
        "next": [
            {"token": _tokenizer.decode([int(t)], clean_up_tokenization_spaces=False), "p": float(p)}
            for p, t in zip(top.values.tolist(), top.indices.tolist())
        ],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/status")
def status() -> JSONResponse:
    return JSONResponse(
        {"status": _state["status"], "detail": _state["detail"], "error": _state["error"]}
    )


@app.post("/run")
def run(req: dict = Body(...)) -> JSONResponse:
    """Compute one slice and return the self-contained page as a string.

    The page goes back in JSON and the browser drops it into iframe.srcdoc. That
    beats a GET /slice/{id} route because it needs no server-side store, no ids
    and no eviction: mode="embed" pages are already fully self-contained, so the
    only thing a store would buy is a URL nobody asked for.
    """
    if _state["status"] != "ready":
        return JSONResponse(
            {"error": f"model is not ready ({_state['status']}: {_state['detail']})"}, 503
        )

    prompt = str(req.get("prompt", "")).strip()
    if not prompt:
        return JSONResponse({"error": "prompt is empty"}, 400)

    def clamp(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(req.get(key, default))))
        except (TypeError, ValueError):
            return default

    max_seq_len = clamp("max_seq_len", 256, 16, MAX_SEQ_CAP)
    layer_stride = clamp("layer_stride", 2, 1, 8)
    top_n = clamp("top_n", 8, 1, TOP_N_CAP)
    last_n = clamp("last_n_tokens", 0, 0, MAX_SEQ_CAP) or None
    gen_tokens = clamp("gen_tokens", 32, 0, GEN_CAP)

    import torch

    from jlens.vis import _template, build_page, compute_slice

    with _lock:
        try:
            # Retry the d3 fetch here, not inside build_page: vis memoises only on
            # success, so a failure there would land after the whole slice compute.
            _template("embed")
            t0 = time.time()
            data = compute_slice(
                _model,
                _lens,
                prompt,
                top_n=top_n,
                layer_stride=layer_stride,
                last_n_tokens=last_n,
                max_seq_len=max_seq_len,
                max_tracked=MAX_TRACKED,
                mask_display=True,
            )
            t_compute = time.time() - t0
            t0 = time.time()
            page, _raw, _payload = build_page(
                data,
                prompt,
                title="J-lens slice",
                description=f"{MODEL_ID} ({PRECISION}), lens {LENS_REPO}",
                mode="embed",
            )
            t_render = time.time() - t0
            t0 = time.time()
            # Outside compute_slice, so no lens hooks are registered; inside the
            # lock, since it shares the GPU. Runs on the ids the slice used, so
            # the continuation follows exactly the prompt the grid shows.
            output = _generate(data.context_token_ids, gen_tokens, top_n) if gen_tokens else None
            t_gen = time.time() - t0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return JSONResponse(
                {"error": "CUDA out of memory. Lower max_seq_len or raise layer_stride, then retry."},
                500,
            )
        except Exception as exc:  # noqa: BLE001 - surface the real thing, do not hide it
            log("RUN FAILED\n" + traceback.format_exc())
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, 500)

    used = len(data.context_token_ids)
    stats = (
        f"{used} tokens used"
        + (f" (truncated at max_seq_len={max_seq_len})" if used == max_seq_len else "")
        + f" · {data.seq_len} positions x {len(data.layers)} rows"
        + f" · forward+readout {t_compute:.1f}s · render {t_render:.1f}s"
        + (f" · generate {t_gen:.1f}s" if output else "")
        + f" · page {len(page) / 1e6:.1f} MB"
    )
    log(stats)
    return JSONResponse({"html": page, "stats": stats, "prompt": prompt, "output": output})


def _compact(r: dict) -> dict:
    """The slice of one flood record the page draws. The full record goes to
    FLOOD_OUT; this is what streams, so it stays small."""
    last = r["positions"][-1]
    # The answer is computed at the token before the last (the "=" or "is"); the
    # last position's band often holds the task instead. Both are drawn.
    prev = r["positions"][-2] if len(r["positions"]) > 1 else None
    pos_rank = [p["band_min_answer_rank"] for p in r["positions"]]
    return {
        "id": r["id"],
        "family": r["family"],
        "tier": r["tier"],
        "correct": r["correct"],
        "answer_later": r["answer_later"],
        "band_scored": r["band_scored"],
        "answer_single_token": r["answer_single_token"],
        "continuation": r["continuation"],
        "model_p": last["model_answer_p"],
        "model_top1": last["model_top1"],
        "model_top1_p": last["model_top1_p"],
        "pos_rank": pos_rank,
        "rank2": min([v for v in pos_rank[-2:] if v is not None], default=None),
        "layer_rank": [l["answer_rank"] for l in last["layers"]],
        "layer_rank2": [l["answer_rank"] for l in prev["layers"]] if prev else None,
        "entropy": last["band_min_entropy"],
        "occ": last["occupancy"],
        "stab": last["top1_stability"],
        "inter": last["band_min_inter_rank"],
    }


def _flood_worker(band: list[int], positions: int, gen: int, items: list[dict]) -> None:
    global _flood_mask
    try:
        import torch

        if _flood_mask is None:
            vocab = int(
                _model.unembed(torch.zeros(_model.d_model, device=_model.input_device)).shape[-1]
            )
            _flood_mask = wordlike_mask(_model, vocab)
        t0 = time.time()
        with open(FLOOD_OUT, "w") as f:
            for item in items:
                with _lock:
                    r = measure(_model, _lens, item, band, positions, gen, _flood_mask)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                with _flood_cv:
                    _flood["results"].append(_compact(r))
                    _flood_cv.notify_all()
        log(f"flood done: {len(items)} items in {time.time() - t0:.0f}s -> {FLOOD_OUT}")
    except Exception:
        _flood["error"] = traceback.format_exc()
        log("FLOOD FAILED\n" + _flood["error"])
    finally:
        with _flood_cv:
            _flood["running"] = False
            _flood_cv.notify_all()


@app.post("/flood/start")
def flood_start(req: dict = Body(...)) -> JSONResponse:
    """Run the whole flood suite in this process, one item at a time."""
    if _state["status"] != "ready":
        return JSONResponse(
            {"error": f"model is not ready ({_state['status']}: {_state['detail']})"}, 503
        )
    if _flood["running"]:
        return JSONResponse({"error": "a flood run is already going"}, 409)

    def clamp(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(req.get(key, default))))
        except (TypeError, ValueError):
            return default

    fitted = _lens.source_layers
    lo = clamp("band_lo", 18, min(fitted), max(fitted))
    hi = clamp("band_hi", 44, min(fitted), max(fitted))
    positions = clamp("positions", 6, 1, 8)
    gen = clamp("gen", 12, 1, 32)
    band = [l for l in fitted if lo <= l <= hi]
    if not band:
        return JSONResponse({"error": f"empty band {lo}..{hi}; fitted layers are {fitted}"}, 400)

    items = [json.loads(l) for l in open(FLOOD_ITEMS) if l.strip()]
    with _flood_cv:
        _flood.update(
            running=True,
            total=len(items),
            results=[],
            error=None,
            params={"band": band, "positions": positions, "gen": gen},
        )
        _flood_cv.notify_all()
    threading.Thread(
        target=_flood_worker, args=(band, positions, gen, items), daemon=True
    ).start()
    log(
        f"flood start: {len(items)} items | band {band[0]}..{band[-1]} ({len(band)} layers) "
        f"| positions {positions} | gen {gen}"
    )
    return JSONResponse({"total": len(items), "params": _flood["params"]})


@app.get("/flood/status")
def flood_status(since: int = 0) -> JSONResponse:
    """Whole state for a late-joining tab, or for curl. The page uses /flood/events."""
    results = _flood["results"]
    return JSONResponse(
        {
            "running": _flood["running"],
            "total": _flood["total"],
            "done": len(results),
            "error": _flood["error"],
            "params": _flood["params"],
            "results": results[since:],
        }
    )


def _sse(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/flood/events")
def flood_events() -> StreamingResponse:
    """Server-sent events: a snapshot, then one `item` per completed item, then
    `done`. A sync generator, so uvicorn runs it on a worker thread and the
    blocking wait on the condition is fine."""

    def gen():
        with _flood_cv:
            sent = len(_flood["results"])
            running = _flood["running"]
            snapshot = {
                "running": running,
                "total": _flood["total"],
                "done": sent,
                "error": _flood["error"],
                "params": _flood["params"],
                "results": list(_flood["results"]),
            }
        yield _sse("snapshot", snapshot)
        while True:
            if not running:
                yield _sse(
                    "done",
                    {"done": sent, "total": snapshot["total"], "error": _flood["error"]},
                )
                return
            with _flood_cv:
                if len(_flood["results"]) == sent and _flood["running"]:
                    _flood_cv.wait(15)
                new = _flood["results"][sent:]
                sent += len(new)
                running = _flood["running"]
                snapshot["total"] = _flood["total"]
            if not new and running:
                yield ": keepalive\n\n"  # the tunnel and any proxy in between
                continue
            for rec in new:
                yield _sse("item", rec)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/flood/results.jsonl")
def flood_results():
    if not os.path.exists(FLOOD_OUT):
        return JSONResponse({"error": "no flood results on disk yet"}, 404)
    return FileResponse(
        FLOOD_OUT, media_type="application/x-ndjson", filename="flood-results.jsonl"
    )


PAGE = """<!doctype html><meta charset=utf-8><title>J-lens</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0d0f12;color:#c9cdd4;
     font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:10px 14px;border-bottom:1px solid #1e2229;display:flex;gap:14px;align-items:baseline}
h1{font-size:13px;margin:0;color:#e6e9ee;font-weight:600;letter-spacing:.04em}
#detail{color:#6b727d;font-size:11px}
main{padding:14px;display:flex;flex-direction:column;gap:10px}
textarea{width:100%;height:64px;resize:vertical;background:#12151a;color:#e6e9ee;
         border:1px solid #262b33;border-radius:3px;padding:8px;font:inherit}
textarea:focus,input:focus{outline:none;border-color:#3d6ea8}
.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
label{color:#6b727d;font-size:11px}
input[type=number]{width:66px;background:#12151a;color:#e6e9ee;border:1px solid #262b33;
                   border-radius:3px;padding:3px 5px;font:inherit}
button{background:#1b2531;color:#cfe0f5;border:1px solid #33465e;border-radius:3px;
       padding:5px 14px;font:inherit;cursor:pointer}
button:hover:enabled{background:#22303f}
button:disabled{opacity:.4;cursor:default}
.ex{background:none;border:none;color:#6f8db3;padding:0;text-align:left;font-size:11px;
    cursor:pointer;text-decoration:underline dotted}
.ex:hover{color:#9dc0e8}
#stats,#flood_stats{color:#6b727d;font-size:11px;min-height:1.5em}
#err,#flood_err{white-space:pre-wrap;color:#e08a7a;background:#1a1214;border:1px solid #3d2226;
     border-radius:3px;padding:8px;font-size:11px;display:none}
#output{display:none;background:#12151a;border:1px solid #262b33;border-radius:3px;padding:8px 10px;
        white-space:pre-wrap;word-break:break-word}
#output .ctx{color:#6b727d}
#output .gen{color:#e6e9ee;background:#1b2531}
#next{color:#6b727d;font-size:11px;margin-top:6px}
#next b{color:#cfe0f5;font-weight:600}
iframe{width:100%;height:78vh;border:1px solid #262b33;border-radius:3px;background:#fff;display:none}
section{border-top:1px solid #1e2229;margin-top:6px;padding-top:12px;
        display:flex;flex-direction:column;gap:8px}
h2{font-size:12px;margin:0;color:#e6e9ee;font-weight:600;letter-spacing:.04em}
a{color:#6f8db3;font-size:11px}
.cap{color:#6b727d;font-size:11px;margin:8px 0 2px}
canvas{display:block}
#flood_vis{display:none}
.sw{display:inline-block;width:9px;height:9px;margin-right:4px;vertical-align:middle}
#tip{position:fixed;display:none;z-index:9;pointer-events:none;max-width:340px;
     white-space:pre-wrap;background:#12151a;border:1px solid #33465e;border-radius:3px;
     padding:5px 7px;font-size:11px;color:#c9cdd4}
</style>
<header><h1>J-LENS</h1><span id=detail>connecting...</span></header>
<main>
<textarea id=prompt spellcheck=false placeholder="prompt"></textarea>
<div class=row id=examples></div>
<div class=row>
  <label>max_seq_len <input type=number id=max_seq_len value=256 min=16 max=512></label>
  <label>layer_stride <input type=number id=layer_stride value=2 min=1 max=8></label>
  <label>top_n <input type=number id=top_n value=8 min=1 max=12></label>
  <label>last_n_tokens <input type=number id=last_n_tokens value=0 min=0 max=512></label>
  <label>gen_tokens <input type=number id=gen_tokens value=32 min=0 max=128></label>
  <button id=go disabled>run</button>
  <span id=stats></span>
</div>
<div id=err></div>
<div id=output><span class=ctx></span><span class=gen></span><div id=next></div></div>
<iframe id=out></iframe>
<section>
  <h2>FLOOD SUITE</h2>
  <div class=row>
    <label>band_lo <input type=number id=band_lo value=18></label>
    <label>band_hi <input type=number id=band_hi value=44></label>
    <label>positions <input type=number id=positions value=6 min=1 max=8></label>
    <label>gen <input type=number id=gen value=12 min=1 max=32></label>
    <button id=flood_go disabled>start flood</button>
    <span id=flood_stats></span>
    <a href="/flood/results.jsonl" download>results.jsonl</a>
  </div>
  <div id=flood_err></div>
  <div id=flood_vis>
    <div class=cap id=cap_acc>accuracy by tier</div>
    <canvas id=cv_acc></canvas>
    <div class=row id=fam_legend></div>
    <div class=cap id=cap_band>answer rank in the band, last position</div>
    <canvas id=cv_band></canvas>
    <div class=cap id=cap_lead>when the answer becomes readable</div>
    <canvas id=cv_lead></canvas>
    <div class=cap>rank scale</div>
    <canvas id=cv_key></canvas>
  </div>
</section>
</main>
<div id=tip></div>
<script>
const EX = __EXAMPLES__;
const $ = id => document.getElementById(id);
EX.forEach(t => {
  const b = document.createElement('button');
  b.className = 'ex'; b.textContent = t;
  b.onclick = () => { $('prompt').value = t; };
  $('examples').appendChild(b);
});
// The final row (L47) is the model's own output, not a lens readout: it is the
// sanity check that the NF4 quantization has not drifted the lens off.
let busy = false;
// Greedy continuation, highlighted after the dimmed prompt, then the top next-token
// probabilities at the last position: the distribution the L47 row is showing top-1 of.
function showOutput(prompt, o) {
  const box = $('output');
  if (!o) { box.style.display = 'none'; return; }
  box.querySelector('.ctx').textContent = prompt;
  box.querySelector('.gen').textContent = o.continuation;
  const next = $('next'); next.textContent = 'next token: ';
  o.next.forEach((n, i) => {
    if (i) next.append('  ');
    const b = document.createElement('b'); b.textContent = JSON.stringify(n.token);
    next.append(b, ' ' + (100 * n.p).toFixed(1) + '%');
  });
  box.style.display = 'block';
}
// --- flood suite ------------------------------------------------------------
// Rows are items in the order they ran, so both heatmaps share a y axis and the
// canvases are sized for the whole suite up front: a new record paints one row
// and never forces a redraw. A snapshot event repaints everything, which is how
// an EventSource reconnect resyncs.
const VIRIDIS = ['#440154','#3b528b','#21918c','#5ec962','#fde725'].map(
  c => [1,3,5].map(i => parseInt(c.slice(i, i + 2), 16)));
const FAMCOL = ['#6ea8e8','#e8a06e','#7ed19a','#d98fd0','#e8d36e','#8fd0d9','#b0b6c0'];
const GUT = 84, PADT = 6, ROWH = 6;
let fbusy = false, frecs = [], fparams = null, ftotal = 0, fes = null;
let fcols = null, fcw = null, fmark = null;
const famIdx = {};

function famColor(f) {
  if (!(f in famIdx)) famIdx[f] = Object.keys(famIdx).length;
  return FAMCOL[famIdx[f] % FAMCOL.length];
}
function ramp(t) {
  t = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(t)), f = t - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return 'rgb(' + a.map((v, k) => Math.round(v + f * (b[k] - v))).join(',') + ')';
}
// rank 0 is the darkest stop; 10k and past it saturate at the brightest.
const rankColor = r => r === null || r === undefined ? '#2b2226' : ramp(Math.log10(r + 1) / 4);

function drawKey() {
  const cv = $('cv_key'), g = cv.getContext('2d');
  cv.width = 300; cv.height = 26;
  for (let x = 0; x < 200; x++) { g.fillStyle = ramp(x / 199); g.fillRect(x, 0, 1, 10); }
  g.font = '10px ui-monospace,monospace'; g.fillStyle = '#6b727d'; g.textBaseline = 'top';
  ['rank 0','10','100','1k','10k+'].forEach((s, i) => {
    g.textAlign = i === 0 ? 'left' : (i === 4 ? 'right' : 'center');
    g.fillText(s, i * 50, 12);
  });
}
function drawFamLegend(fams) {
  const box = $('fam_legend'); box.textContent = '';
  fams.forEach(f => {
    const sw = document.createElement('span');
    sw.className = 'sw'; sw.style.background = famColor(f);
    const lab = document.createElement('label');
    lab.append(sw, f);
    box.appendChild(lab);
  });
}
function drawAcc() {
  const cv = $('cv_acc'), g = cv.getContext('2d');
  const W = cv.width = 480, H = cv.height = 150, X0 = 32, Y0 = 10;
  const PW = W - X0 - 12, PH = H - Y0 - 24;
  g.clearRect(0, 0, W, H);
  const by = {}, tiers = new Set();
  frecs.forEach(r => {
    const f = by[r.family] = by[r.family] || {};
    f[r.tier] = f[r.tier] || [0, 0];
    f[r.tier][0] += r.correct ? 1 : 0; f[r.tier][1]++;
    tiers.add(r.tier);
  });
  const ts = [...tiers].sort((a, b) => a - b);
  if (!ts.length) return;
  const tmin = ts[0], tmax = ts[ts.length - 1];
  const X = t => X0 + (tmax > tmin ? (t - tmin) / (tmax - tmin) : 0.5) * PW;
  const Y = a => Y0 + (1 - a) * PH;
  g.font = '10px ui-monospace,monospace';
  g.textBaseline = 'middle'; g.textAlign = 'right';
  [0, 0.5, 1].forEach(a => {
    g.strokeStyle = '#1e2229'; g.beginPath();
    g.moveTo(X0, Y(a) + 0.5); g.lineTo(X0 + PW, Y(a) + 0.5); g.stroke();
    g.fillStyle = '#6b727d'; g.fillText(a.toFixed(1), X0 - 5, Y(a));
  });
  g.textBaseline = 'top'; g.textAlign = 'center';
  g.fillStyle = '#6b727d';
  ts.forEach(t => g.fillText('t' + t, X(t), Y0 + PH + 5));
  Object.keys(by).forEach(f => {
    const pts = Object.keys(by[f]).map(Number).sort((a, b) => a - b)
      .map(t => [X(t), Y(by[f][t][0] / by[f][t][1])]);
    g.strokeStyle = g.fillStyle = famColor(f); g.lineWidth = 1.5;
    g.beginPath();
    pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]));
    g.stroke();
    pts.forEach(p => { g.beginPath(); g.arc(p[0], p[1], 2, 0, 6.284); g.fill(); });
  });
  drawFamLegend(Object.keys(by));
}
function drawHeatRow(cv, i, ranks, cw, mark) {
  const g = cv.getContext('2d'), rec = frecs[i], y = PADT + i * ROWH, w = ranks.length * cw;
  // items.jsonl interleaves nth-letter with count-letter, so a separator on every
  // family change would be a line per row. The rail carries the grouping; the name
  // and the rule are drawn once, where the family first appears.
  const st = fmark[cv.id];
  g.fillStyle = famColor(rec.family);
  g.fillRect(GUT - 7, y, 4, ROWH - 1);
  if (!st.seen.has(rec.family)) {
    st.seen.add(rec.family);
    g.strokeStyle = '#39404a'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(GUT - 8, y - 0.5); g.lineTo(GUT + w, y - 0.5); g.stroke();
    st.lastY = Math.max(y, st.lastY + 11);  // 6px rows, 10px text: never overlap
    g.font = '10px ui-monospace,monospace'; g.textAlign = 'left'; g.textBaseline = 'top';
    g.fillText(rec.family, 2, st.lastY);
  }
  if (!rec.band_scored || !rec.answer_single_token) {
    // rank is meaningless here: single-letter answers, or answers the tokenizer
    // splits. Hatched rather than dropped, so the row order still lines up.
    g.save(); g.beginPath(); g.rect(GUT, y, w, ROWH - 1); g.clip();
    g.fillStyle = '#23272e'; g.fillRect(GUT, y, w, ROWH - 1);
    g.strokeStyle = '#3c434d'; g.lineWidth = 1; g.beginPath();
    for (let hx = GUT - ROWH; hx < GUT + w; hx += 5) {
      g.moveTo(hx, y + ROWH - 1); g.lineTo(hx + ROWH, y);
    }
    g.stroke(); g.restore();
  } else {
    for (let c = 0; c < ranks.length; c++) {
      g.fillStyle = rankColor(ranks[c]);
      g.fillRect(GUT + c * cw, y, cw, ROWH - 1);
    }
  }
  if (mark) {
    g.fillStyle = rec.correct ? '#5fbf7a' : (rec.answer_later ? '#d6a34a' : '#c9564c');
    g.fillRect(GUT + w + 4, y, 2, ROWH - 1);
  }
}
// Per layer, the better of the last two positions: the answer is prepared at the
// token before the last, so reading only the last one calls that absence.
function bandRanks(rec) {
  const a = rec.layer_rank, b = rec.layer_rank2;
  if (!b) return a;
  return a.map((v, i) => v === null || v === undefined ? b[i]
    : (b[i] === null || b[i] === undefined ? v : Math.min(v, b[i])));
}
function drawRow(i) {
  drawHeatRow($('cv_band'), i, bandRanks(frecs[i]), fcw.band, true);
  drawHeatRow($('cv_lead'), i, frecs[i].pos_rank, fcw.lead, false);
}
function floodProgress(state) {
  $('flood_stats').textContent = ftotal ? frecs.length + ' / ' + ftotal + ' · ' + state : '';
}
function floodSetup() {
  const vis = $('flood_vis');
  if (!ftotal || !fparams || !fparams.band) { vis.style.display = 'none'; return; }
  vis.style.display = 'block';
  fcols = {band: fparams.band, lead: []};
  for (let i = fparams.positions; i > 0; i--) fcols.lead.push('-' + i);
  fcw = {band: Math.max(4, Math.min(24, Math.floor(420 / fcols.band.length))),
         lead: Math.max(6, Math.min(26, Math.floor(300 / fcols.lead.length)))};
  fmark = {cv_band: {seen: new Set(), lastY: -99}, cv_lead: {seen: new Set(), lastY: -99}};
  const b = $('cv_band'), c = $('cv_lead'), h = PADT + ftotal * ROWH + 6;
  b.width = GUT + fcols.band.length * fcw.band + 10; b.height = h;
  c.width = GUT + fcols.lead.length * fcw.lead + 4; c.height = h;
  // A lens fitted on sparse source_layers gives a band whose columns are evenly
  // spaced in pixels but NOT in model depth, so neighbouring columns are not
  // neighbouring layers. Say so, or the heatmap reads as a contiguous stack.
  // Inert for a densely fitted lens: the note only appears when layers are skipped.
  const bd = fcols.band;
  const dense = bd[bd.length - 1] - bd[0] + 1 === bd.length;
  $('cap_band').textContent = 'answer rank in the band, best of last two positions · L'
    + bd[0] + '..L' + bd[bd.length - 1]
    + (dense ? '' : ' (' + bd.length + ' fitted layers, unevenly spaced'
        + ' - columns are not consecutive layers; hover for the layer)')
    + ' · right edge: green correct, amber later, red wrong';
  $('cap_lead').textContent = 'when the answer becomes readable · positions '
    + fcols.lead[0] + '..' + fcols.lead[fcols.lead.length - 1];
  drawKey();
}
function floodAdd(rec) { frecs.push(rec); drawRow(frecs.length - 1); drawAcc(); }
function floodErr(msg) { $('flood_err').style.display = 'block'; $('flood_err').textContent = msg; }

function floodOpen() {
  if (fes) fes.close();
  fes = new EventSource('/flood/events');
  fes.addEventListener('snapshot', e => {
    const s = JSON.parse(e.data);
    frecs = []; fparams = s.params; ftotal = s.total;
    floodSetup();
    s.results.forEach(floodAdd);
    if (s.error) floodErr(s.error);
    fbusy = s.running;
    if (s.running) $('flood_go').disabled = true;  // re-enabling is poll()'s job
    floodProgress(s.running ? 'running' : (s.error ? 'error' : 'done'));
  });
  fes.addEventListener('item', e => { floodAdd(JSON.parse(e.data)); floodProgress('running'); });
  fes.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    if (d.error) floodErr(d.error);
    floodProgress(d.error ? 'error' : 'done');
    fbusy = false; $('flood_go').disabled = false;
    fes.close(); fes = null;
  });
}
['band','lead'].forEach(k => {
  const cv = $(k === 'band' ? 'cv_band' : 'cv_lead');
  cv.onmousemove = ev => {
    if (!fcols) return;
    const box = cv.getBoundingClientRect(), cols = fcols[k];
    const row = Math.floor((ev.clientY - box.top - PADT) / ROWH);
    const col = Math.floor((ev.clientX - box.left - GUT) / fcw[k]);
    if (row < 0 || row >= frecs.length || col < 0 || col >= cols.length) {
      $('tip').style.display = 'none'; return;
    }
    const rec = frecs[row];
    const skipped = rec.band_scored && rec.answer_single_token ? '' : '  (not band-scored)';
    const fr = v => v === null || v === undefined ? 'n/a' : v;
    const line = k === 'band'
      ? `layer ${cols[col]}   rank -1 ${fr(rec.layer_rank[col])}, `
        + `-2 ${fr(rec.layer_rank2 && rec.layer_rank2[col])}${skipped}`
      : `position ${cols[col]}   rank ${fr(rec.pos_rank[col])}${skipped}`;
    $('tip').textContent = `${rec.id}  ${rec.family} tier ${rec.tier}
${line}
${JSON.stringify(rec.continuation)}`;
    $('tip').style.display = 'block';
    $('tip').style.left = (ev.clientX + 12) + 'px';
    $('tip').style.top = (ev.clientY + 12) + 'px';
  };
  cv.onmouseleave = () => { $('tip').style.display = 'none'; };
});
$('flood_go').onclick = async () => {
  fbusy = true; $('flood_go').disabled = true; $('flood_err').style.display = 'none';
  $('flood_stats').textContent = 'starting...';
  const body = {};
  for (const k of ['band_lo','band_hi','positions','gen']) body[k] = parseInt($(k).value, 10);
  try {
    const r = await fetch('/flood/start', {method:'POST', headers:{'Content-Type':'application/json'},
                                           body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok || d.error) {
      $('flood_stats').textContent = ''; floodErr(d.error || ('HTTP ' + r.status));
      fbusy = false; $('flood_go').disabled = false; return;
    }
    floodOpen();
  } catch (e) {
    $('flood_stats').textContent = ''; floodErr(String(e));
    fbusy = false; $('flood_go').disabled = false;
  }
};
floodOpen();  // picks up a run already in progress, e.g. after a page reload

async function poll() {
  let delay = 3000;
  try {
    const s = await (await fetch('/status')).json();
    $('detail').textContent = s.status === 'ready' ? s.detail : s.status + ': ' + s.detail;
    if (!busy) $('go').disabled = s.status !== 'ready';
    if (!fbusy) $('flood_go').disabled = s.status !== 'ready';
    if (s.status === 'failed') { $('err').style.display = 'block'; $('err').textContent = s.error; }
    if (s.status !== 'loading') delay = 15000;  // keep watching: bootstrap.sh is re-runnable
  } catch (e) { $('detail').textContent = 'server unreachable'; }
  setTimeout(poll, delay);
}
poll();
$('go').onclick = async () => {
  busy = true; $('go').disabled = true; $('err').style.display = 'none';
  $('stats').textContent = 'running...';
  const body = {prompt: $('prompt').value};
  for (const k of ['max_seq_len','layer_stride','top_n','last_n_tokens','gen_tokens'])
    body[k] = parseInt($(k).value, 10);
  try {
    const r = await fetch('/run', {method:'POST', headers:{'Content-Type':'application/json'},
                                   body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok || d.error) {
      $('stats').textContent = '';
      $('err').style.display = 'block'; $('err').textContent = d.error || ('HTTP ' + r.status);
    } else {
      $('stats').textContent = d.stats;
      showOutput(d.prompt, d.output);
      $('out').style.display = 'block'; $('out').srcdoc = d.html;
    }
  } catch (e) {
    $('stats').textContent = '';
    $('err').style.display = 'block'; $('err').textContent = String(e);
  }
  busy = false; $('go').disabled = false;
};
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # json.dumps, not html.escape: this lands inside a <script> block, where HTML
    # entities are not decoded. The </ guard keeps a literal from closing the tag.
    examples = json.dumps(EXAMPLES).replace("</", "<\\/")
    return HTMLResponse(PAGE.replace("__EXAMPLES__", examples))
