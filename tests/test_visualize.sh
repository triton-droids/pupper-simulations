#!/usr/bin/env bash
# Test suite for visualize.sh — no SSH connectivity required.
# All tests use --dry-run or provoke early errors.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VISUALIZE="$PROJECT_DIR/visualize.sh"

PASS=0
FAIL=0
TOTAL=0

# ── Helpers ────────────────────────────────────────────────────────────
run_test() {
  local name="$1"
  TOTAL=$((TOTAL + 1))
  echo -n "  [$TOTAL] $name ... "
}

pass() {
  PASS=$((PASS + 1))
  echo "PASS"
}

fail() {
  FAIL=$((FAIL + 1))
  echo "FAIL"
  [ -n "${1:-}" ] && echo "       -> $1"
}

# Create a temporary .env for tests that need one
make_test_env() {
  cat > "$PROJECT_DIR/.env.test" <<'EOF'
SSH_KEY_PATH="/tmp/test-ssh-key-visualize"
SSH_DIRECTORY="testuser/test-project"
DROIDS_IP_ADDRESS="testuser@192.168.1.100"
EOF
}

cleanup() {
  rm -f "$PROJECT_DIR/.env.test"
  rm -f /tmp/test-ssh-key-visualize
}
trap cleanup EXIT

# ── Tests ──────────────────────────────────────────────────────────────
echo ""
echo "Running visualize.sh tests..."
echo ""

# 1. Missing .env → clear error
run_test "Missing .env produces error"
# Run with a non-existent .env by temporarily hiding the real one
output=$(cd /tmp && bash "$VISUALIZE" 2>&1 || true)
if echo "$output" | grep -q "Error: .env file not found"; then
  pass
else
  fail "Expected '.env file not found' error, got: $output"
fi

# 2. Missing required env var → names the missing variable
run_test "Missing required env var names the variable"
cat > "$PROJECT_DIR/.env.test" <<'EOF'
SSH_KEY_PATH="/tmp/test-ssh-key-visualize"
SSH_DIRECTORY=""
DROIDS_IP_ADDRESS="testuser@192.168.1.100"
EOF
# Temporarily swap .env
mv_env=false
if [ -f "$PROJECT_DIR/.env" ]; then
  mv "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.bak"
  mv_env=true
fi
cp "$PROJECT_DIR/.env.test" "$PROJECT_DIR/.env"
output=$(bash "$VISUALIZE" 2>&1 || true)
if echo "$output" | grep -q "SSH_DIRECTORY"; then
  pass
else
  fail "Expected SSH_DIRECTORY in error, got: $output"
fi
rm -f "$PROJECT_DIR/.env"
if [ "$mv_env" = true ]; then
  mv "$PROJECT_DIR/.env.bak" "$PROJECT_DIR/.env"
fi

# 3. Invalid SSH key path → error before scp
run_test "Invalid SSH key path produces error"
make_test_env
if [ -f "$PROJECT_DIR/.env" ]; then
  mv "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.bak"
  mv_env=true
else
  mv_env=false
fi
cp "$PROJECT_DIR/.env.test" "$PROJECT_DIR/.env"
output=$(bash "$VISUALIZE" 2>&1 || true)
if echo "$output" | grep -q "Error: SSH key not found"; then
  pass
else
  fail "Expected 'SSH key not found' error, got: $output"
fi
rm -f "$PROJECT_DIR/.env"
if [ "$mv_env" = true ]; then
  mv "$PROJECT_DIR/.env.bak" "$PROJECT_DIR/.env"
fi

# 4. Unknown flag → usage message
run_test "Unknown flag shows usage"
make_test_env
if [ -f "$PROJECT_DIR/.env" ]; then
  mv "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.bak"
  mv_env=true
else
  mv_env=false
fi
cp "$PROJECT_DIR/.env.test" "$PROJECT_DIR/.env"
output=$(bash "$VISUALIZE" --bogus-flag 2>&1 || true)
if echo "$output" | grep -q "Unknown flag" && echo "$output" | grep -q "Usage"; then
  pass
else
  fail "Expected usage message with unknown flag error, got: $output"
fi
rm -f "$PROJECT_DIR/.env"
if [ "$mv_env" = true ]; then
  mv "$PROJECT_DIR/.env.bak" "$PROJECT_DIR/.env"
fi

# 5. --dry-run prints correct scp command
run_test "--dry-run prints scp command with correct remote path"
make_test_env
touch /tmp/test-ssh-key-visualize
if [ -f "$PROJECT_DIR/.env" ]; then
  mv "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.bak"
  mv_env=true
else
  mv_env=false
fi
cp "$PROJECT_DIR/.env.test" "$PROJECT_DIR/.env"
output=$(bash "$VISUALIZE" --dry-run 2>&1)
if echo "$output" | grep -q "scp.*testuser@192.168.1.100:testuser/test-project/locomotion/outputs/policy.onnx"; then
  pass
else
  fail "Expected scp command with correct remote path, got: $output"
fi
rm -f "$PROJECT_DIR/.env"
if [ "$mv_env" = true ]; then
  mv "$PROJECT_DIR/.env.bak" "$PROJECT_DIR/.env"
fi

# 6. --dry-run --video prints both policy and video scp commands
run_test "--dry-run --video prints both scp commands"
make_test_env
touch /tmp/test-ssh-key-visualize
if [ -f "$PROJECT_DIR/.env" ]; then
  mv "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.bak"
  mv_env=true
else
  mv_env=false
fi
cp "$PROJECT_DIR/.env.test" "$PROJECT_DIR/.env"
output=$(bash "$VISUALIZE" --dry-run --video 2>&1)
policy_scp=$(echo "$output" | grep -c "scp.*policy.onnx" || true)
video_scp=$(echo "$output" | grep -c "scp.*latest_video.mp4" || true)
if [ "$policy_scp" -ge 1 ] && [ "$video_scp" -ge 1 ]; then
  pass
else
  fail "Expected both policy and video scp commands, got: $output"
fi
rm -f "$PROJECT_DIR/.env"
if [ "$mv_env" = true ]; then
  mv "$PROJECT_DIR/.env.bak" "$PROJECT_DIR/.env"
fi

# 7. --dry-run --download-only does not print mjpython command
run_test "--dry-run --download-only omits mjpython"
make_test_env
touch /tmp/test-ssh-key-visualize
if [ -f "$PROJECT_DIR/.env" ]; then
  mv "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.bak"
  mv_env=true
else
  mv_env=false
fi
cp "$PROJECT_DIR/.env.test" "$PROJECT_DIR/.env"
output=$(bash "$VISUALIZE" --dry-run --download-only 2>&1)
if echo "$output" | grep -q "mjpython"; then
  fail "mjpython should not appear with --download-only, got: $output"
else
  pass
fi
rm -f "$PROJECT_DIR/.env"
if [ "$mv_env" = true ]; then
  mv "$PROJECT_DIR/.env.bak" "$PROJECT_DIR/.env"
fi

# 8. Local output directory creation works
run_test "Creates locomotion/outputs/ directory"
make_test_env
touch /tmp/test-ssh-key-visualize
if [ -f "$PROJECT_DIR/.env" ]; then
  mv "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.bak"
  mv_env=true
else
  mv_env=false
fi
cp "$PROJECT_DIR/.env.test" "$PROJECT_DIR/.env"
# Remove the directory if it exists (but preserve .gitkeep)
bash "$VISUALIZE" --dry-run > /dev/null 2>&1
if [ -d "$PROJECT_DIR/locomotion/outputs" ]; then
  pass
else
  fail "locomotion/outputs/ directory was not created"
fi
rm -f "$PROJECT_DIR/.env"
if [ "$mv_env" = true ]; then
  mv "$PROJECT_DIR/.env.bak" "$PROJECT_DIR/.env"
fi

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS/$TOTAL passed, $FAIL failed"
echo ""
if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
