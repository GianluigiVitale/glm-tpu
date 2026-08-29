#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
REFUSED REJECTED_OBSERVER_PERTURBATION: the layer-1 RMS-input callback path is
tombstoned. Protected run greenfield_legacy_layer1_rms_input_p8155_20260829T192233297523063Z
reproduced DB551 exactly (557434 position and 573438 score mismatches, first at
event 1). Do not acquire, retry, or promote callback-derived operands.
EOF
exit 2
