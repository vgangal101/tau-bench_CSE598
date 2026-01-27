# Quick Test Script - 4B Models Only

This is a lightweight test to validate the combined experiment approach before running the full experiment.

## Why This Test?

- **Faster to schedule**: Only needs 1 GPU (vs 2 GPUs for full experiment)
- **Lower memory**: 32GB RAM (vs 96GB for full experiment)
- **Quicker runtime**: ~20-30 minutes total (vs 1-2 hours for full)
- **Validates approach**: Confirms the combined workflow works before committing to larger job

## What It Does

1. Starts **both** vLLM servers on **1 GPU** (using memory sharing):
   - User Simulator: Qwen3-4B on port 8000
   - Agent: Qwen3-4B on port 8001
2. Runs a **single quick experiment**:
   - Environment: retail only
   - Strategy: tool-calling only
   - Tasks: First 2 tasks only
3. Validates that everything works end-to-end

## How to Run

```bash
cd SOL_env/day1
sbatch combined_test_4b.sh
```

## Monitor Progress

```bash
# Watch job queue
watch -n 10 'squeue -u $USER'

# Once running, watch output
tail -f logs/test_4b_*.out

# Check for errors
tail -f logs/test_4b_*.err
```

## Expected Timeline

- **Queue wait**: 5-30 minutes (1 GPU easier to get than 2)
- **Server startup**: 5-10 minutes (model download + initialization)
- **Experiment run**: 5-10 minutes (only 2 tasks)
- **Total**: ~20-45 minutes

## Resource Usage

```
CPUs: 8
GPUs: 1x A100
RAM: 32GB
Time limit: 2 hours
```

## Success Indicators

If the test works, you'll see:
```
✓ TEST PASSED!
Status: SUCCESS
```

And results will be saved to:
```
SOL_env/day1/test_results/retail/tool-calling/
```

## If It Fails

Check the logs:
```bash
# Main experiment log
cat logs/test_4b_*.err

# Server logs
tail -100 logs/user_4b_*.log
tail -100 logs/agent_4b_*.log
```

Common issues:
- **OOM (Out of Memory)**: GPU memory exhausted - this would mean 1 GPU can't handle both models
- **Port conflicts**: Another job using ports 8000/8001 (unlikely on fresh node)
- **Model download failure**: Network issue, will retry automatically

## Next Steps

### If Test Succeeds ✓
Your combined approach works! You can:
1. **Cancel the full 2-GPU job** if it's still pending:
   ```bash
   scancel 46155916
   ```
2. **Wait for results** from this test to validate quality
3. **Run full experiment** with confidence

### If Test Fails ✗
1. Check error logs to diagnose issue
2. Consider using the **original 3-job workflow** instead (more reliable for large models)
3. Ask for help with specific error messages

## Differences from Full Experiment

| Aspect | This Test | Full Experiment |
|--------|-----------|-----------------|
| User model | Qwen3-4B | Qwen2.5-32B |
| Agent model | Qwen3-4B | Qwen3-4B |
| GPUs | 1 | 2 |
| Memory | 32GB | 96GB |
| Environments | retail only | retail + airline |
| Strategies | tool-calling only | tool-calling + act + react |
| Tasks per env | 2 | 3 |
| Total experiments | 1 | 6 |
| Runtime | ~20-45 min | ~1-2 hours |

## Key Learning

If both 4B models can share 1 GPU successfully, it proves:
- ✅ The combined script approach works
- ✅ Health checks work correctly
- ✅ Experiments can connect to servers
- ✅ Background process management works

However, the **full experiment still needs 2 GPUs** because Qwen2.5-32B (~40GB VRAM) + Qwen3-4B (~10GB VRAM) won't fit on one A100 (40GB).
