# strix-halo-testing

WIP Documentation here: https://llm-tracker.info/_TOORG/Strix-Halo

## ROCm environment scripts

Use `rocm-env.sh` for a standard ROCm install in `/opt/rocm`. For nightly ROCm builds from TheRock set up with the path `/home/lhl/therock/rocm-7.0`, source `rocm-therock-env.sh`:

```bash
source ./rocm-therock-env.sh
```

This exports HIP and ROCm variables similar to the provided fish configuration.
