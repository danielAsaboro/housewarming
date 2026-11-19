#!/usr/bin/env bash
# Build, strip pipeline working files from the bundle, deploy to Vercel production.
set -euo pipefail
cd "$(dirname "$0")/.."
pnpm build >/dev/null
# ship only the rooms listed in index.json
LISTED=$(node -e "console.log(require('./public/rooms/index.json').rooms.join(' '))")
for d in dist/rooms/*/; do id=$(basename "$d"); case " $LISTED " in *" $id "*) ;; *) rm -r "$d";; esac; done
find dist/rooms \( -name 'raw.ply' -o -name 'aligned.ply' -o -name 'state.json' -o -name 'pipeline.log' -o -name '*.zip' -o -name 'raw-*.glb' -o -name 'pano.jpg' \) -delete
find dist/rooms -type d -name 'urdf' -prune -exec rm -r {} +
OUT="${DEPLOY_DIR:-$TMPDIR/housewarming-deploy}/housewarming"
mkdir -p "$OUT" && rsync -a --delete --exclude .vercel dist/ "$OUT/"
[ -d .vercel ] && mkdir -p "$OUT/.vercel" && cp .vercel/project.json "$OUT/.vercel/" 2>/dev/null || true
du -sh "$OUT"
cd "$OUT" && vercel deploy --prod --yes 2>&1 | grep -E 'Aliased|Production|Error' || true
