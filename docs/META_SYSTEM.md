# Sistema meta (historial cruzado)

Los runs de `scripts/laser_target_match.py` pueden registrarse al finalizar en `runs/_meta/history.sqlite` mediante `scripts/meta/recorder.py` (`record_experiment`). Si falla, se emite un warning y la corrida no se aborta.

## CLI

```bash
python -m scripts.meta.analyzer --last 10
python -m scripts.meta.analyzer --compare 3 7
python -m scripts.meta.analyzer --regressions
python -m scripts.meta.analyzer --baseline-for <sha256_hex_input>
python -m scripts.meta.proposer --input ruta/foto.png --target ruta/ref.png
```

## Consultas SQL utiles

```sql
-- Ultimos 20 mejores scores
SELECT id, finished_at, best_score, best_pixel_error, preprocess_mode
FROM experiments ORDER BY id DESC LIMIT 20;

-- Mejor score por hash de target
SELECT target_hash, MIN(best_score) AS best, COUNT(*) AS n
FROM experiments GROUP BY target_hash ORDER BY best ASC;

-- Top candidatos agregados (param_stats)
SELECT experiment_id, rank_in_run, score, param_value
FROM param_stats ORDER BY experiment_id DESC, rank_in_run ASC LIMIT 50;
```

## Optuna

`scripts/laser_optuna_search.py` genera `optuna.db`, `optuna_top.json` y `optuna_match_seed.sqlite` compatible con `--from-db` de `laser_target_match.py`.

## Dashboard

```bash
pip install -e ".[dashboard]"
streamlit run scripts/dashboard.py
```
