#!/usr/bin/env bash

configdir=${1%/}; shift
logdir=${1%/}; shift

for config in $configdir/*; do
    uuid="$(python -c "import uuid; print(str(uuid.uuid4()).lower())")"
    echo "sbatch" \
        "--partition=gpu" \
        "--gres=gpu:1" \
        "--error \"$logdir/$uuid.err\"" \
        "--output \"$logdir/$uuid.out\"" \
        "--exclude=ai0[1-6],artemis[1-7],falcon[1-10],granger[1-8],hermes[1-4],lynx[01-12],nibbler[1-4],slurm[1-5],trillian[1-3]" \
        "./scripts/slurm_run.sh \"$config\" \"$uuid\""
    sbatch \
        --partition=gpu \
        --gres=gpu:1 \
        --error "$logdir/$uuid.err" \
        --output "$logdir/$uuid.out" \
        --exclude=ai0[1-6],artemis[1-7],falcon[1-10],granger[1-8],hermes[1-4],lynx[01-12],nibbler[1-4],slurm[1-5],trillian[1-3] \
        ./scripts/slurm_run.sh "$config" "$uuid"
done