#!/bin/sh
# Fake ssh/scp for prometheus-slurm tests.
#
# The production client wraps every spawn in a 200ms query budget. A Python
# interpreter under parallel `cargo test --workspace` can miss that budget
# before it prints a job id, so submit returns QueryTimeout and never records
# the job name. This script speaks the same protocol and starts in a few
# milliseconds.
set -u

if [ -z "${PROMETHEUS_SLURM_MOCK_DIR:-}" ]; then
    printf 'PROMETHEUS_SLURM_MOCK_DIR unset\n' >&2
    exit 2
fi
MOCK=$PROMETHEUS_SLURM_MOCK_DIR
mkdir -p "$MOCK"

prog=$(basename "$0")

join_args() {
    line=
    sep=
    for w in "$@"; do
        line="${line}${sep}${w}"
        sep=' '
    done
    printf '%s' "$line"
}

log_remote() {
    host=$1
    shift
    remote=$(join_args "$@")
    printf '%s\n' "$remote" >>"$MOCK/remote.log"
    printf '{"host":"%s","remote":"%s","prog":"%s"}\n' "$host" "$remote" "$prog" >>"$MOCK/commands.jsonl"
}

if [ "$prog" = "scp" ]; then
    log_remote "" "$@"
    exit 0
fi

# Skip ssh flags the way OpenSSH does, then the next word is the host.
while [ $# -gt 0 ]; do
    a=$1
    if [ "$a" = "--" ]; then
        shift
        break
    fi
    case $a in
        -b|-c|-D|-E|-e|-F|-I|-i|-J|-L|-l|-m|-O|-o|-p|-Q|-R|-S|-W|-w)
            shift
            [ $# -gt 0 ] && shift
            continue
            ;;
        -*)
            shift
            continue
            ;;
        *)
            break
            ;;
    esac
done

host=${1:-}
if [ $# -gt 0 ]; then
    shift
fi
log_remote "$host" "$@"

if [ $# -eq 0 ]; then
    exit 0
fi

cmd=
for w in "$@"; do
    case $w in
        sbatch|squeue|sacct|scancel)
            cmd=$w
            break
            ;;
    esac
done

read_trim() {
    if [ ! -f "$1" ]; then
        return 0
    fi
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$1" | head -n 1
}

maybe_hang() {
    kind=$1
    flag=$MOCK/hang_queries
    if [ ! -f "$flag" ]; then
        return 0
    fi
    kinds=$(read_trim "$flag")
    case $kinds in
        ""|1|true|all) ;;
        *)
            case ",$kinds," in
                *",$kind,"*) ;;
                *) return 0 ;;
            esac
            ;;
    esac
    sleep 3
}

snapshot_state() {
    state=${PROMETHEUS_SLURM_STATE_DIR:-}
    if [ -z "$state" ] || [ ! -d "$state" ]; then
        printf 'no state dir\n' >"$MOCK/state_at_sbatch.missing"
        return 0
    fi
    rm -rf "$MOCK/state_at_sbatch"
    cp -a "$state" "$MOCK/state_at_sbatch"
}

job_id_from() {
    prev=
    lastnum=
    for w in "$@"; do
        case $prev in
            -j|--job|--jobid)
                printf '%s' "$w"
                return 0
                ;;
        esac
        case $w in
            --job=*|--jobid=*)
                printf '%s' "${w#*=}"
                return 0
                ;;
        esac
        case $w in
            *[!0-9]*|"") ;;
            *) lastnum=$w ;;
        esac
        prev=$w
    done
    printf '%s' "$lastnum"
}

case $cmd in
    sbatch)
        snapshot_state
        mode=$(read_trim "$MOCK/sbatch_mode")
        [ -n "$mode" ] || mode=ok
        if [ "$mode" = "hang" ]; then
            sleep 3
        fi
        if [ "$mode" = "fail" ]; then
            printf 'sbatch: simulated failure\n' >&2
            exit 1
        fi
        if [ "$mode" = "panic" ]; then
            printf 'sbatch: simulated panic\n' >&2
            kill -ABRT $$
            exit 99
        fi
        jobid=$(read_trim "$MOCK/sbatch_jobid")
        [ -n "$jobid" ] || jobid=4242
        printf '%s\n' "$jobid"
        exit 0
        ;;
    squeue)
        maybe_hang squeue
        if [ -f "$MOCK/squeue_out" ]; then
            cat "$MOCK/squeue_out"
        fi
        exit 0
        ;;
    sacct)
        maybe_hang sacct
        jid=$(job_id_from "$@")
        if [ -n "$jid" ] && [ -f "$MOCK/states/$jid" ]; then
            printf '%s\n' "$(read_trim "$MOCK/states/$jid")"
            exit 0
        fi
        if [ -f "$MOCK/default_state" ]; then
            printf '%s\n' "$(read_trim "$MOCK/default_state")"
        fi
        exit 0
        ;;
    scancel)
        remote=$(join_args "$@")
        printf '%s\n' "$remote" >>"$MOCK/scancel.log"
        for w in "$@"; do
            case $w in
                -u|--user|--user=*)
                    printf 'yes\n' >"$MOCK/scancel_u"
                    ;;
                *[!0-9]*|"") ;;
                *)
                    printf '%s\n' "$w" >>"$MOCK/cancelled_ids"
                    ;;
            esac
        done
        exit 0
        ;;
esac
exit 0
