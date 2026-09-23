# deploy: J-lens dashboard on a rented GPU

> **Branch `model/gemma-4-12b-it`** — serves `google/gemma-4-12B-it`. `main` serves
> `Qwen/Qwen3.8-27B`. One model per branch; see `MODEL-BRANCHES.md` on `main`.

Rent a GPU on vast.ai, serve the Jacobian lens dashboard on it, reach it from your
laptop over an SSH tunnel at `http://localhost:7860`.

Four scripts, in order:

| script | runs on | what it does |
|---|---|---|
| `provision.sh` | laptop | picks an offer, creates the instance, waits for it |
| `bootstrap.sh` | instance | installs deps, downloads model and lens, starts the server |
| `tunnel.sh` | laptop | forwards `localhost:7860` to the instance |
| `destroy.sh` | laptop | kills the instance. This is the only thing that stops billing |
| `flood.sh` | laptop | runs `experiments/flood` on the instance headless and pulls the results back. Optional |
| `wait.sh` | laptop | when no offer matches: retries provision.sh on an interval, waits for bootstrap, opens the tunnel |
| `apply.sh` | laptop | copies the local `deploy/`, `experiments/`, `jlens/` to the instance and restarts the dashboard. `--pull` does a git pull there instead |

`dashboard.py` is the server: a prompt box that renders jlens' own slice
visualisation. Above the slice it shows the model's own output: a greedy
continuation of `gen_tokens` tokens (default 32, `0` to skip) decoded from the same
token ids the slice used, and the top-`top_n` next-token probabilities at the last
prompt position. The L47 row shows the top-1 of that distribution, masked to
word-like tokens; the panel shows it unmasked, with probabilities. It binds to
`127.0.0.1` on the instance, so it is reachable through the tunnel and invisible to
the public internet.

**No model weights are ever downloaded to your laptop.** The 24GB checkpoint and the
703MiB lens are fetched by the instance, from the instance. Your laptop only moves
JSON and HTML over SSH.

## Why this model and this lens

Fitting a lens costs real H100-hours, so we apply a published one rather than fit our
own. The lens for `google/gemma-4-12B-it` is at `lamm-mit/gemma4-jacobian-lenses`, file
`gemma4-12b-it/paper/seed0.pt`. The app log prints the lens's own `n_prompts` when it
loads.

**This branch serves bf16, and that is the point of it.** 12B in bf16 is ~24GB of
weights plus ~1.5GB of lens, which fits the same 48GB card class the project already
rents. `main` cannot do this: Qwen3.8-27B in bf16 is 56GB and needs an 80GB card that is
outside the budget, which is why `main` quantizes to NF4 and carries an uncorrected
drift between a bf16-fitted lens and an NF4-served model. **Here there is no
quantization drift to caveat.** The lens was fitted in bf16 and is read against a bf16
model.

This page describes `model/gemma-4-12b-it`. **Serving a different model is a different
git branch, not a different set of environment variables** — the VRAM floors, the offer
filter, the flood band defaults and the final-row convention all move with the model.
[`MODEL-BRANCHES.md`](MODEL-BRANCHES.md) is the policy: what a model branch may change,
how it stays rebased on `main`, and the checklist for adding one.

The built-in check is still the **final row, L47**. This model has 48 layers, so layer
47 is not in the lens, jlens skips transport for it, and that row is the model's true
output. If L47 looks sane and the rows below it do not, suspect the lens before the
model.

`JLENS_PRECISION=nf4` (~18GB of VRAM) and `int8` (~22GB) are still available and open up
much cheaper cards, at the cost of reintroducing the drift `main` has to live with. On a
card too small for the precision you asked for, the bootstrap refuses to start before it
downloads anything.

**What is different about this lens.** It has **25 fitted layers, and they are sparse**:
`0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39, 41, 43,
45`, transporting to a target layer of 46. `main`'s Qwen lens has 63 contiguous layers.
Two consequences worth holding on to:

- The flood band defaults here are `18..44`, not `main`'s `24..58`. Both are the same
  ~38-92% slice of depth; the absolute numbers move with the layer count.
- A band is a *subset of the fitted layers*, not a contiguous range of model layers. Ask
  for `18..44` and you get the 14 fitted layers inside it, unevenly spaced. The middle
  visual's columns are those 14, so neighbouring columns are not neighbouring layers.

## Usage

First time only: the instance clones this repo from GitHub, so `deploy/` must be
committed and pushed to a public fork before you provision.

```bash
git add deploy/*.sh deploy/dashboard.py deploy/README.md
git commit -m "deploy scripts" && git push
```

`provision.sh` checks for this and refuses to spend money if the push is missing.
The paths are listed explicitly because `deploy/.instance` and `deploy/.known_hosts` are
runtime state, and this fork is public.

Then:

```bash
cd deploy

./provision.sh          # search, show the top offers, confirm, create, wait
./tunnel.sh             # forward localhost:7860, leave this running
# open http://localhost:7860 in a browser
./destroy.sh            # when finished. This stops the billing
```

`provision.sh` returns once sshd is up, but the instance is still installing and
downloading. **Nothing listens on port 7860 until that finishes**, so for the first 8 to
20 minutes there is no dashboard to load: a browser pointed at `localhost:7860` gets a
connection error, not a loading page. Watch the setup instead, and open the tunnel once
the log prints `server started`.

```bash
./tunnel.sh --logs      # then Ctrl-C, and ./tunnel.sh
```

`tunnel.sh` also warns you when the remote port is not listening yet. Once the server is
up, the page does report its own load status while the model loads, which is a further 4
to 8 minutes.

Other things you will want:

```bash
./provision.sh --status     # state, uptime, accrued cost of the current instance
./provision.sh --offer N    # use a specific offer id instead of the cheapest match
./provision.sh --yes        # skip the confirmation prompt

./tunnel.sh --logs          # tail the setup and app logs on the instance
./tunnel.sh --port 7861     # use a different local port
./tunnel.sh <id>            # target an instance other than the one in .instance

./destroy.sh --list         # every instance on the account, and what it costs
./destroy.sh <id>           # destroy one by id
```

The instance id is written to `deploy/.instance` as soon as the instance exists. The
scripts read it from there, so you rarely need to pass an id by hand.

### Changing precision

`provision.sh` does not pass `JLENS_PRECISION` to the instance, so a fresh instance is
always `bf16` on this branch. To change it, SSH in and re-run the bootstrap. It is
idempotent: it
skips the install and the downloads, and restarts the server.

```bash
JLENS_PRECISION=nf4 bash /workspace/jacobian-lens/deploy/bootstrap.sh
```

It refuses to start if the card is too small for the precision you asked for, before
downloading anything.

## Running the flood suite from the dashboard

The `FLOOD SUITE` section at the bottom of the page runs `experiments/flood` **inside
the dashboard process**, reusing the model that is already loaded. There is no second
copy of the weights, and the page stays usable: the run takes the slice lock one item
at a time, so a `/run` slice interleaves between items instead of waiting for all 361.

Set `band_lo`/`band_hi` (clamped to the lens's fitted layers), `positions` (the last N
prompt positions to read out, 1-8) and `gen` (greedy tokens for correctness, 1-32), then
press `start flood`. Rows stream in over server-sent events and the three visuals fill in
as each item finishes:

| visual | rows | columns | colour |
|---|---|---|---|
| accuracy by tier | one line per family | tier | the family's own colour |
| answer rank in the band, last position | items, in the order they ran | the band's fitted layers | `log10(rank+1)`, dark = rank 0 |
| when the answer becomes readable | the same items | the read positions, earliest to last | same scale: this is the `lead` picture |

Hover any cell for the item id, the layer or position, the rank and the continuation.
A hatched row is one whose rank is meaningless (`band_scored: false`, or an answer the
tokenizer does not encode as a single token); the marker on the right edge of the middle
visual is green for correct, amber for answer-later, red for wrong. The left rail colours
each row by family, because `items.jsonl` interleaves `nth-letter` with `count-letter`.

The full per-layer records go to `JLENS_FLOOD_OUT` (default `/workspace/flood-results.jsonl`),
rewritten at the start of every run. Download them with the `results.jsonl` link next to
the button, or `curl localhost:7860/flood/results.jsonl`, then run
`python experiments/flood/analyze.py <file>` on the laptop.

`flood.sh` is still the headless alternative: it stops the dashboard, runs `run.py` over
SSH, pulls the results back and analyzes them. Use it for an unattended sweep; use the
page when you want to watch the tier curves form.

## Cost

Credit at time of writing: **$18.47**. Design target: stay under $15.

**The rates below have not been re-measured for this branch.** They are `main`'s
numbers, for a 120GB disk and `gpu_ram >= 45`. This branch rents a 60GB disk and accepts
`gpu_ram >= 40`, so both the storage component and the card mix differ — expect the real
figures to come in at or below these. Run `./provision.sh` and read the printed table;
it is the only current source. They move daily in any case.

| card | VRAM | $/hr with storage |
|---|---|---|
| RTX A6000 | 48GB | 0.433 |
| L40S | 45GB | 0.567 |
| RTX 4090 | 48GB | 0.578 |

Cheaper cards exist. An A40 at $0.281/hr shows up on a looser search. The filter in
`provision.sh` excludes some of them on purpose: it requires at least 16GB of usable host
RAM, Ampere or newer, an allow-listed GPU name, decent download speed, and cheap
bandwidth. A host that charges $20/TB for downloads adds about half a dollar to the 25GB
of model and lens here — a smaller penalty than on `main`, where the same host costs over
a dollar against 59GB.

At $0.433/hr:

| | time | cost |
|---|---|---|
| one-off setup, per instance | 8 to 20 min | $0.06 to $0.14 |
| setup plus an hour of use | ~1.25 hr | ~$0.54 |
| the $15 target buys | ~34 hr | $15 |
| all $18.47 of credit buys | ~42 hr | $18.47 |

Setup is not the risk. Forgetting to destroy is. Storage bills for the whole life of
the instance, running or stopped, so **stopping an instance does not stop the burn.
Only `destroy.sh` does.**

Guardrails already in the scripts:

- `provision.sh` prints the rate and the runway and asks before creating anything.
- It refuses to run if `deploy/.instance` still points at a live instance.
- It refuses to run if credit is below $2. Change with `JLENS_CREDIT_FLOOR`.
- If it fails or you interrupt it after creation, it shouts the instance id and the
  destroy command on the way out. If the create call itself fails after vast.ai accepted
  it, it re-reads the account for an instance labelled `jlens` created in the last
  minute, records the id, and prints the same warning.
- `destroy.sh` re-queries after the delete and does not claim success until the
  contract is actually released.
- After a successful destroy it lists any other instance still running on the
  account. That is the check that catches a forgotten instance from a past session.

None of that survives your laptop. Closing the lid or killing the tunnel leaves the
instance billing, so running `destroy.sh` is mandatory, not advisory. The instance does
carry an idle self-destruct, but it is inert until you hand it an API key:

```bash
scp -P <ssh-port> ~/.config/vastai/vast_api_key root@<ssh-host>:/workspace/.jlens/vast_api_key
```

`tunnel.sh` prints the host and port. Armed, it destroys the instance after two hours
with no dashboard traffic. To change the window, re-run the bootstrap with
`JLENS_IDLE_KILL_S=<seconds>`.

If you are ever unsure whether something is running:

```bash
./destroy.sh --list
```

That reads the account, not the local state file, so it sees instances this repo never
knew about.

## What your laptop needs

- `jq`, `curl`, `ssh`, `git`.
- The vast.ai CLI at `~/.local/bin/vastai`. Override with `VASTAI=/path/to/vastai`.
  Only used for offer search.
- A vast.ai API key at `~/.config/vastai/vast_api_key`, or in `VAST_API_KEY`. The
  scripts read it at point of use and never print it. `provision.sh` and `destroy.sh`
  pass it to curl on stdin; `tunnel.sh` still passes it in an argument, so it is briefly
  visible in `ps` on your own machine.
- An SSH key at `~/.ssh/id_ed25519`, already registered on the vast.ai account.
  Override with `JLENS_SSH_KEY`. Do not run bare `vastai create ssh-key`: with no
  argument it generates a new keypair and overwrites your existing one.
- This fork pushed public on `main`, with `deploy/` in it.

No GPU, no CUDA, no torch, no model weights. Nothing heavy is installed locally.

`tunnel.sh` keeps its own `deploy/.known_hosts`. vast.ai recycles IP and port pairs
across machines, so a changed host key is normal churn here. If SSH refuses on a
changed key, delete that file and try again.

## Troubleshooting

**Browser says connection refused or reset.** The server has not started yet. Expected
for the first 8 to 20 minutes: see the note under Usage. `./tunnel.sh --logs` shows
which stage is running.

**Setup seems stuck.** Watch it: `./tunnel.sh --logs`. The two long stages are the model
download and the load onto the card, and both print timings. The full bootstrap transcript is
at `/var/log/portal/jlens-bootstrap.log` on the instance. `provision.sh --status` shows
what vast.ai thinks the instance is doing.

**Instance never reaches `running`.** If `actual_status` reaches `exited`, `unknown`
or `offline`, it will never progress. `provision.sh` detects this and stops. Destroy
it and provision again, which picks a different host.

**Gated HF repo.** Both repos are public today and no token is needed. If that
changes, the download fails with a 403 and `GatedRepoError`. Accept the terms on the
model page while logged in, create a read token, then SSH in and re-run the bootstrap
with `HF_TOKEN=hf_... bash /workspace/jacobian-lens/deploy/bootstrap.sh`. Pass the
token over SSH, never in the provision script: the onstart script is stored on
vast.ai's side.

**CUDA out of memory.** The dashboard catches this and says so rather than dying.
Lower `max_seq_len` or raise `layer_stride` in the form and run again. The expensive
combination is 512 tokens at stride 1, which is two full-vocab sorts on every one of
48 rows. If it happens at the defaults, the card is smaller than the offer claimed:
check the VRAM line in the app log.

**Tunnel port in use.** `tunnel.sh` checks the port before connecting and tells you.
Use `./tunnel.sh --port 7861` and browse `localhost:7861`. The remote port stays 7860.

**Tunnel drops.** Ctrl-C and re-run it. The tunnel is stateless and the instance keeps
running, along with its billing. Closing the tunnel does not stop the instance.

**Dashboard returns 503.** The server is up and the model is still loading. The header
shows the current stage. If it says `failed`, the page prints the full traceback.

**Page is huge or slow to render.** The visualisation inlines its own data, so page
size scales with tokens by rows by tracked tokens. Tracked tokens are capped at 256.
Raise `layer_stride` to halve the rows.
