#!/usr/bin/env bash
# Smoke-test a running service: health, a real prediction, and input validation.
# Usage: scripts/smoke_test.sh [base_url]
set -euo pipefail
URL="${1:-http://127.0.0.1:8000}"

for _ in $(seq 1 30); do
  curl -sf "$URL/health" >/dev/null && break
  sleep 1
done
curl -sf "$URL/health" | python3 -c 'import json,sys; assert json.load(sys.stdin)["status"] == "ok"'

good='{"revenue": 60000, "loan_amnt": 12000, "fico_n": 690, "dti_n": 18, "emp_length": "5 years", "purpose": "debt_consolidation", "home_ownership_n": "RENT", "addr_state": "NY"}'
curl -sf -X POST -H 'Content-Type: application/json' -d "$good" "$URL/predict" | tee /dev/stderr \
  | python3 -c 'import json,sys; b=json.load(sys.stdin); assert 0 <= b["default_probability"] <= 1 and b["decision"] in ("approve", "reject"), b'
echo >&2

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{"revenue": -1}' "$URL/predict")
[ "$code" = "422" ] || { echo "expected 422 for bad input, got $code"; exit 1; }

echo "Smoke test passed for $URL"
