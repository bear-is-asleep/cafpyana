DAY=checkpoint8_full
# ncpu <= chunk-size; larger chunks = fewer det-var loops (RAM scales ~linearly with chunk-size)
OPTS_DET="--ncpu 16 --chunk-size 40 --recompute-norm --cut all_cont"
OPTS_COSMIC="--ncpu 16 --chunk-size 20 --recompute-norm"
OPTS_SLIM="--ncpu 16 --chunk-size 20 --recompute-norm"

python scripts/build_detsys_universes.py --full-det    --day "$DAY" $OPTS_DET
#python scripts/build_detsys_universes.py --full-cosmic  --day "$DAY" $OPTS_COSMIC
python scripts/build_detsys_universes.py --full-slim    --day "$DAY" $OPTS_SLIM
