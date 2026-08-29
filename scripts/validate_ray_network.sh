#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in
  firewall)
    expected_target=${2:?expected target tag is required}
    pod_id=${3:?pod id is required}
    mapfile -t firewall_lines
    if [[ ${#firewall_lines[@]} -ne 1 ]] ||
      [[ $(awk -F '\t' '{ print NF }' <<<"${firewall_lines[0]}") -ne 7 ]]; then
      echo "invalid or stale Ray firewall contract" >&2
      exit 1
    fi
    firewall_record=${firewall_lines[0]}
    IFS=$'\t' read -r network direction priority sources allowed disabled target \
      <<<"$firewall_record"
    [[ $pod_id =~ ^[0-9]+$ &&
       $expected_target =~ ^tpu-[A-Za-z0-9_-]+-${pod_id}$ &&
       $network == default && $direction == INGRESS &&
       $priority == 1000 && $sources == 192.168.0.0/16 &&
       $allowed == tcp:1024-65535 && $disabled == False &&
       $target == "$expected_target" ]] || {
      echo "invalid or stale Ray firewall contract" >&2
      exit 1
    }
    ;;
  receipts)
    marker=${2:?receipt marker is required}
    expected_count=${3:?expected count is required}
    expected_owners=${4:-}
    expected_fields=${5:-2}
    [[ $marker =~ ^[A-Z0-9_]+$ && $expected_count =~ ^[1-9][0-9]*$ &&
       $expected_fields =~ ^([2-9]|1[0-6])$ ]] || {
      echo "invalid receipt-validation arguments" >&2
      exit 2
    }
    payload=$(cat)
    if ! owners=$(awk -v marker="$marker" -v fields="$expected_fields" '
      $1 == marker {
        if (NF != fields || $2 !~ /^[A-Za-z0-9_.-]+$/) bad = 1
        else print $2
      }
      END { if (bad) exit 1 }
    ' <<<"$payload"); then
      echo "malformed $marker receipt" >&2
      exit 1
    fi
    total=$(awk 'NF { count++ } END { print count + 0 }' <<<"$owners")
    unique=$(printf '%s\n' "$owners" | awk 'NF' | sort -u | wc -l)
    actual_owners=$(printf '%s\n' "$owners" | awk 'NF' | sort -V -u | paste -sd, -)
    [[ $total -eq $expected_count && $unique -eq $expected_count &&
       ( -z $expected_owners || $actual_owners == "$expected_owners" ) ]] || {
      echo "incomplete or duplicate $marker receipts" >&2
      exit 1
    }
    ;;
  bounded)
    seconds=${2:?timeout seconds are required}
    shift 2
    [[ $seconds =~ ^[1-9][0-9]*$ && $# -gt 0 ]] || {
      echo "invalid bounded-command arguments" >&2
      exit 2
    }
    exec timeout --signal=TERM --kill-after=10 -- "$seconds" "$@"
    ;;
  *)
    echo "usage: $0 firewall EXPECTED_TARGET POD_ID | receipts MARKER COUNT [OWNERS [FIELDS]] | bounded SECONDS COMMAND..." >&2
    exit 2
    ;;
esac
