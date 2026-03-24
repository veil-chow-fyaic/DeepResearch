#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

USER_MODEL="${MODEL:-}"
USER_MODEL_PATH="${MODEL_PATH:-}"
USER_DATASET="${DATASET:-}"
USER_OUTPUT_PATH="${OUTPUT_PATH:-}"
USER_ROLLOUT_COUNT="${ROLLOUT_COUNT:-}"
USER_TEMPERATURE="${TEMPERATURE:-}"
USER_PRESENCE_PENALTY="${PRESENCE_PENALTY:-}"
USER_MAX_WORKERS="${MAX_WORKERS:-}"
USER_MAX_LLM_CALL_PER_RUN="${MAX_LLM_CALL_PER_RUN:-}"
USER_DEEP_RESEARCH_MIN_ROUNDS="${DEEP_RESEARCH_MIN_ROUNDS:-}"
USER_DEEP_RESEARCH_MIN_TOOL_CALLS="${DEEP_RESEARCH_MIN_TOOL_CALLS:-}"
USER_DEEP_RESEARCH_MIN_SEARCH_CALLS="${DEEP_RESEARCH_MIN_SEARCH_CALLS:-}"
USER_DEEP_RESEARCH_MIN_VISIT_CALLS="${DEEP_RESEARCH_MIN_VISIT_CALLS:-}"
USER_DEEP_RESEARCH_MIN_SCHOLAR_CALLS="${DEEP_RESEARCH_MIN_SCHOLAR_CALLS:-}"
USER_DEEP_RESEARCH_REFLECTION_INTERVAL="${DEEP_RESEARCH_REFLECTION_INTERVAL:-}"
USER_DEEP_RESEARCH_MAX_MINUTES="${DEEP_RESEARCH_MAX_MINUTES:-}"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a  # automatically export all variables
source "$ENV_FILE"
set +a  # stop automatically exporting

[ -n "$USER_MODEL" ] && MODEL="$USER_MODEL"
[ -n "$USER_MODEL_PATH" ] && MODEL_PATH="$USER_MODEL_PATH"
[ -n "$USER_DATASET" ] && DATASET="$USER_DATASET"
[ -n "$USER_OUTPUT_PATH" ] && OUTPUT_PATH="$USER_OUTPUT_PATH"
[ -n "$USER_ROLLOUT_COUNT" ] && ROLLOUT_COUNT="$USER_ROLLOUT_COUNT"
[ -n "$USER_TEMPERATURE" ] && TEMPERATURE="$USER_TEMPERATURE"
[ -n "$USER_PRESENCE_PENALTY" ] && PRESENCE_PENALTY="$USER_PRESENCE_PENALTY"
[ -n "$USER_MAX_WORKERS" ] && MAX_WORKERS="$USER_MAX_WORKERS"
[ -n "$USER_MAX_LLM_CALL_PER_RUN" ] && MAX_LLM_CALL_PER_RUN="$USER_MAX_LLM_CALL_PER_RUN"
[ -n "$USER_DEEP_RESEARCH_MIN_ROUNDS" ] && DEEP_RESEARCH_MIN_ROUNDS="$USER_DEEP_RESEARCH_MIN_ROUNDS"
[ -n "$USER_DEEP_RESEARCH_MIN_TOOL_CALLS" ] && DEEP_RESEARCH_MIN_TOOL_CALLS="$USER_DEEP_RESEARCH_MIN_TOOL_CALLS"
[ -n "$USER_DEEP_RESEARCH_MIN_SEARCH_CALLS" ] && DEEP_RESEARCH_MIN_SEARCH_CALLS="$USER_DEEP_RESEARCH_MIN_SEARCH_CALLS"
[ -n "$USER_DEEP_RESEARCH_MIN_VISIT_CALLS" ] && DEEP_RESEARCH_MIN_VISIT_CALLS="$USER_DEEP_RESEARCH_MIN_VISIT_CALLS"
[ -n "$USER_DEEP_RESEARCH_MIN_SCHOLAR_CALLS" ] && DEEP_RESEARCH_MIN_SCHOLAR_CALLS="$USER_DEEP_RESEARCH_MIN_SCHOLAR_CALLS"
[ -n "$USER_DEEP_RESEARCH_REFLECTION_INTERVAL" ] && DEEP_RESEARCH_REFLECTION_INTERVAL="$USER_DEEP_RESEARCH_REFLECTION_INTERVAL"
[ -n "$USER_DEEP_RESEARCH_MAX_MINUTES" ] && DEEP_RESEARCH_MAX_MINUTES="$USER_DEEP_RESEARCH_MAX_MINUTES"

USE_OPENROUTER=false
if [ -n "$OPENROUTER_API_KEY" ] && [ -n "$OPENROUTER_BASE_URL" ] && [ -n "$MODEL" ]; then
    USE_OPENROUTER=true
fi

if [ "$USE_OPENROUTER" = "false" ] && { [ "$MODEL_PATH" = "/your/model/path" ] || [ -z "$MODEL_PATH" ]; }; then
    echo "Error: MODEL_PATH not configured in .env file"
    echo "Hint: if you want to use OpenRouter, set OPENROUTER_API_KEY / OPENROUTER_BASE_URL / MODEL in .env"
    exit 1
fi

MAX_WORKERS="${MAX_WORKERS:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.1}"
ROLLOUT_COUNT="${ROLLOUT_COUNT:-1}"
DEEP_RESEARCH_MIN_ROUNDS="${DEEP_RESEARCH_MIN_ROUNDS:-8}"
DEEP_RESEARCH_MIN_TOOL_CALLS="${DEEP_RESEARCH_MIN_TOOL_CALLS:-8}"
DEEP_RESEARCH_MIN_SEARCH_CALLS="${DEEP_RESEARCH_MIN_SEARCH_CALLS:-3}"
DEEP_RESEARCH_MIN_VISIT_CALLS="${DEEP_RESEARCH_MIN_VISIT_CALLS:-3}"
DEEP_RESEARCH_MIN_SCHOLAR_CALLS="${DEEP_RESEARCH_MIN_SCHOLAR_CALLS:-0}"
DEEP_RESEARCH_REFLECTION_INTERVAL="${DEEP_RESEARCH_REFLECTION_INTERVAL:-3}"
DEEP_RESEARCH_MAX_MINUTES="${DEEP_RESEARCH_MAX_MINUTES:-150}"

resolve_path() {
    local input_path="$1"
    if [ -z "$input_path" ]; then
        return
    fi
    if [[ "$input_path" = /* ]]; then
        printf '%s\n' "$input_path"
    else
        printf '%s\n' "$PROJECT_ROOT/$input_path"
    fi
}

DATASET="$(resolve_path "$DATASET")"
OUTPUT_PATH="$(resolve_path "$OUTPUT_PATH")"

######################################
### 1. start server           ###
######################################

if [ "$USE_OPENROUTER" = "false" ]; then
    echo "Starting VLLM servers..."
    CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6001 --disable-log-requests &
    CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6002 --disable-log-requests &
    CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6003 --disable-log-requests &
    CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6004 --disable-log-requests &
    CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6005 --disable-log-requests &
    CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6006 --disable-log-requests &
    CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6007 --disable-log-requests &
    CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6008 --disable-log-requests &
else
    echo "OpenRouter mode detected, skip local VLLM startup."
fi

#######################################################
### 2. Waiting for the server port to be ready  ###
######################################################

if [ "$USE_OPENROUTER" = "false" ]; then
    timeout=6000
    start_time=$(date +%s)

    main_ports=(6001 6002 6003 6004 6005 6006 6007 6008)
    echo "Mode: All ports used as main model"

    declare -A server_status
    for port in "${main_ports[@]}"; do
        server_status[$port]=false
    done

    echo "Waiting for servers to start..."

    while true; do
        all_ready=true

        for port in "${main_ports[@]}"; do
            if [ "${server_status[$port]}" = "false" ]; then
                if curl -s -f http://localhost:$port/v1/models > /dev/null 2>&1; then
                    echo "Main model server (port $port) is ready!"
                    server_status[$port]=true
                else
                    all_ready=false
                fi
            fi
        done

        if [ "$all_ready" = "true" ]; then
            echo "All servers are ready for inference!"
            break
        fi

        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -gt $timeout ]; then
            echo -e "\nError: Server startup timeout after ${timeout} seconds"

            for port in "${main_ports[@]}"; do
                if [ "${server_status[$port]}" = "false" ]; then
                    echo "Main model server (port $port) failed to start"
                fi
            done

            exit 1
        fi

        printf 'Waiting for servers to start .....'
        sleep 10
    done

    failed_servers=()
    for port in "${main_ports[@]}"; do
        if [ "${server_status[$port]}" = "false" ]; then
            failed_servers+=($port)
        fi
    done

    if [ ${#failed_servers[@]} -gt 0 ]; then
        echo "Error: The following servers failed to start: ${failed_servers[*]}"
        exit 1
    else
        echo "All required servers are running successfully!"
    fi
fi

#####################################
### 3. start infer               ####
#####################################

echo "==== start infer... ===="


cd "$( dirname -- "${BASH_SOURCE[0]}" )"

if [ "$USE_OPENROUTER" = "true" ]; then
    RUN_MODEL="$MODEL"
else
    RUN_MODEL="$MODEL_PATH"
fi

python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model "$RUN_MODEL" --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT --min_rounds $DEEP_RESEARCH_MIN_ROUNDS --min_tool_calls $DEEP_RESEARCH_MIN_TOOL_CALLS --min_search_calls $DEEP_RESEARCH_MIN_SEARCH_CALLS --min_visit_calls $DEEP_RESEARCH_MIN_VISIT_CALLS --min_scholar_calls $DEEP_RESEARCH_MIN_SCHOLAR_CALLS --reflection_interval $DEEP_RESEARCH_REFLECTION_INTERVAL --max_minutes $DEEP_RESEARCH_MAX_MINUTES
