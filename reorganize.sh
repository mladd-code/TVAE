#!/usr/bin/env bash
# reorganize.sh — one-time reorganization of the TVAE repo
# Run from repo root: bash reorganize.sh
# Undo (before committing): git reset --hard HEAD && git clean -fd

set -euo pipefail

mv_safe() {
  local src="$1" dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    echo "  $src -> $dst"
  fi
}

echo "== Directory tree =="
mkdir -p src/tvae/models simulations applications paper \
         results/logs data/raw data/processed \
         archive/diagnostics archive/data_pull future_work/gkx_app
touch results/.gitkeep data/raw/.gitkeep data/processed/.gitkeep

echo "== Core library =="
mv_safe tvae_sim.py     src/tvae/dgp.py
mv_safe static_ae.py    src/tvae/models/static_ae.py
mv_safe tvae_global.py  src/tvae/models/tvae_global.py
mv_safe tvae_local.py   src/tvae/models/tvae_local.py
mv_safe local_pca.py    src/tvae/models/lpca.py

echo "== Simulations =="
mv_safe run_montecarlo.py         simulations/montecarlo_small.py
mv_safe run_final_comparison.py   simulations/montecarlo_final.py
mv_safe run_capacity_sweep.py     simulations/capacity_sweep_stage1.py
mv_safe run_capacity_sweep_v2.py  simulations/capacity_sweep_stage2.py
mv_safe run_scaling.py            simulations/scaling.py
mv_safe scaling_results_v2.py     simulations/scaling_v2.py
mv_safe run_diagnostics.py        simulations/diagnostics.py
mv_safe run_diagnostics_v2.py     simulations/diagnostics_v2.py
mv_safe crossval_lpca.py          simulations/crossval_lpca.py
mv_safe crossval_compare.py       simulations/crossval_compare.py

echo "== Applications =="
mv_safe reconstruction_r2.py                applications/sp100_returns.py
mv_safe reconstruction_r2_vol.py            applications/sp100_vol.py
mv_safe reconstruction_r2_crypto.py         applications/crypto_returns.py
mv_safe reconstruction_r2_crypto_vol.py     applications/crypto_vol.py
mv_safe reconstruction_r2_crypto_k1sweep.py applications/crypto_k1_sweep.py
mv_safe reconstruction_r2_crypto_tau.py     applications/crypto_tau.py
mv_safe bootstrap_recon_crypto.py           applications/bootstrap_crypto_returns.py
mv_safe bootstrap_recon_crypto_vol.py       applications/bootstrap_crypto_vol.py
mv_safe build_vol_panel.py                  applications/build_sp100_vol.py
mv_safe build_crypto_vol_panel.py           applications/build_crypto_vol.py
mv_safe compute_rv.py                       applications/build_commodity_rv.py
mv_safe mvp_app.py                          applications/mvp.py

echo "== GKX / predictive-R^2 app (not part of the paper) =="
mv_safe gkx.py                future_work/gkx_app/gkx.py
mv_safe run_app1_slice.py     future_work/gkx_app/run_app1_slice.py
mv_safe run_tvaec.py          future_work/gkx_app/run_tvaec.py
mv_safe tvaec_app1.py         future_work/gkx_app/tvaec_app1.py
mv_safe tvae_global_app_1.py  future_work/gkx_app/tvae_global_app_1.py
mv_safe tvaec_mvp.py          future_work/gkx_app/tvaec_mvp.py

echo "== Archive: diagnostics and data pulls =="
for f in diag_ae.py diag_factors.py diag_tvaec.py \
         diag_crypto_fields.py diag_crypto_rics.py diag_crypto_rics_v2.py \
         diag_nasdaq_codes.py diag_screener.py diag_panel.py \
         run_step2.py run_step3.py run_step4.py run_step5.py; do
  mv_safe "$f" "archive/diagnostics/$f"
done
for f in refinitiv_pull.py pull_daily.py pull_crypto.py \
         fix_dates.py characteristics.py; do
  mv_safe "$f" "archive/data_pull/$f"
done

echo "== Rewriting import paths =="
# Core renames in every script under simulations/ and applications/
sed -i.bak \
  -e 's/^from tvae_sim import/from tvae.dgp import/' \
  -e 's/^from local_pca import/from tvae.models.lpca import/' \
  -e 's/^from static_ae import/from tvae.models.static_ae import/' \
  -e 's/^from tvae_global import/from tvae.models.tvae_global import/' \
  -e 's/^from tvae_local import/from tvae.models.tvae_local import/' \
  $(find simulations applications -name '*.py') 2>/dev/null || true

# Sibling renames inside applications/
sed -i.bak 's/^from reconstruction_r2 import/from sp100_returns import/' \
  applications/sp100_vol.py 2>/dev/null || true
sed -i.bak 's/^from reconstruction_r2_crypto import/from crypto_returns import/' \
  applications/crypto_vol.py applications/bootstrap_crypto_returns.py \
  applications/crypto_k1_sweep.py applications/crypto_tau.py 2>/dev/null || true
sed -i.bak 's/^from reconstruction_r2_crypto_vol import/from crypto_vol import/' \
  applications/bootstrap_crypto_vol.py 2>/dev/null || true

find . -name '*.py.bak' -delete

echo "== Done =="
echo "Next: add the 8 new files, then 'pip install -e .' and 'git add -A'."