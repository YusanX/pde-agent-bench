RESULTS_DIR="/Users/yusan/agent/pdebench/results/mini-swe-agent"

for case_dir in "$RESULTS_DIR"/*/; do

    case_name=$(basename "$case_dir")
    prompt_file="$case_dir/prompt.md"

    if [ ! -f "$prompt_file" ]; then
        echo " prompt.md not found"
        continue
    fi
    
    cd "$case_dir" || continue
    mini -m "anthropic.claude-sonnet-4-5-20250929-v1:0" -t "$(cat prompt.md)" -y --exit-immediately
    cd - > /dev/null

done

