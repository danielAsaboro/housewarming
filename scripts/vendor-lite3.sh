#!/usr/bin/env bash
# Vendors DEEP Robotics' Lite3 MJCF (deep_robotics_model, BSD-3) and walking policy (Lite3_rl_deploy, BSD-3).
# Robot-only copy: the model's demo floor + light are removed so it can be <include>d into a room scene.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_SHA=$(gh api repos/DeepRoboticsLab/deep_robotics_model/commits/HEAD --jq .sha)
POLICY_SHA=$(gh api repos/DeepRoboticsLab/Lite3_rl_deploy/commits/HEAD --jq .sha)
R=public/robots/lite3; P=public/policy/lite3
mkdir -p "$R/meshes" "$P"
B="https://raw.githubusercontent.com/DeepRoboticsLab/deep_robotics_model/$MODEL_SHA"
curl -sfL "$B/Lite3/mjcf/Lite3.xml" -o "$R/Lite3.upstream.xml"
for m in hip shank thigh torso; do curl -sfL "$B/Lite3/mjcf/meshes/$m.STL" -o "$R/meshes/$m.STL"; done
curl -sfL "$B/LICENSE.txt" -o "$R/LICENSE.txt"
grep -v -E '<geom name="floor"|<light directional' "$R/Lite3.upstream.xml" > "$R/lite3.xml"
Q="https://raw.githubusercontent.com/DeepRoboticsLab/Lite3_rl_deploy/$POLICY_SHA"
curl -sfL "$Q/policy/ppo/policy.onnx" -o "$P/policy.onnx"
curl -sfL "$Q/LICENSE" -o "$P/LICENSE"
printf 'deep_robotics_model %s\nLite3_rl_deploy %s\n' "$MODEL_SHA" "$POLICY_SHA" > public/robots/SOURCES.txt
echo "vendored: model $MODEL_SHA policy $POLICY_SHA"
