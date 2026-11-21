#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

fail() {
    printf '%s\n' "$*" >&2
    exit 2
}

lines=10
order=up
directory=

while (($#)); do
    case $1 in
        -l|--lines)
            (($# >= 2)) || fail "Missing value for $1"
            lines=$2
            shift 2
            ;;
        -o|--order)
            (($# >= 2)) || fail "Missing value for $1"
            order=$2
            shift 2
            ;;
        --)
            shift
            (($# == 1)) && [[ -z $directory ]] || fail 'Expected one log directory'
            directory=$1
            break
            ;;
        -*)
            fail "Unknown option: $1"
            ;;
        *)
            [[ -z $directory ]] || fail 'Expected one log directory'
            directory=$1
            shift
            ;;
    esac
done

[[ $lines =~ ^[0-9]+$ ]] || fail 'The line limit must be a nonnegative integer'
[[ $order == up || $order == down ]] || fail 'Order must be up or down'
[[ -n $directory && -d $directory ]] || fail 'Expected an existing log directory'
[[ $directory == /* ]] || directory=./$directory

select_lines() {
    if [[ $order == up ]]; then
        # Consume the whole stream so sort can finish even with a small limit.
        awk -v limit="$lines" 'NR <= limit'
    else
        tail -n "$lines"
    fi
}

find "$directory" -type f -name '*.log' -exec awk '
    BEGIN {
        prefix = "^\\[ (INFO|DEBUG|WARNING|ERROR) \\][[:space:]][0-9][0-9]:[0-9][0-9]:[0-9][0-9][0-9][0-9][[:space:]]"
        record = prefix "[0-9]+[[:space:]][A-Za-z_]*$"
    }
    {
        # Accept CRLF files while preserving a CR used as a field separator.
        if ($0 !~ record) {
            sub(/\r$/, "")
        }
        if ($0 ~ record) {
            id = $0
            sub(prefix, "", id)
            sub(/[[:space:]].*$/, "", id)
            printf "%s\t%s\n", id, $0
        }
    }
' {} + |
    sort -s -t $'\t' -k1,1nr |
    select_lines |
    cut -f2-
