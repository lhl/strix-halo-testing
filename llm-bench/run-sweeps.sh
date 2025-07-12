### Dense Models
# python llama-cpp-bencher.py -m /models/gguf/llama-2-7b.Q4_0.gguf
python llama-cpp-bencher.py -m /models/gguf/llama-2-7b.Q4_K_M.gguf
python llama-cpp-bencher.py -m /models/gguf/shisa-v2-llama3.1-8b.i1-Q4_K_M.gguf
python llama-cpp-bencher.py -m /models/gguf/Mistral-Small-3.1-24B-Instruct-2503-UD-Q4_K_XL.gguf
python llama-cpp-bencher.py -m /models/gguf/gemma-3-27b-it-UD-Q4_K_XL.gguf
python llama-cpp-bencher.py -m /models/gguf/Qwen3-32B-Q8_0.gguf
python llama-cpp-bencher.py -m /models/gguf/shisa-v2-llama3.3-70b.i1-Q4_K_M.gguf
python llama-cpp-bencher.py -m /models/gguf/python llama-cpp-bencher.py -m /models/gguf/CohereForAI_c4ai-command-a-03-2025-Q5_K_M-00001-of-00002.gguf

### MoEs
python llama-cpp-bencher.py --moe -m /models/gguf/Qwen3-30B-A3B-UD-Q4_K_XL.gguf
python llama-cpp-bencher.py --moe -m /models/gguf/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf
python llama-cpp-bencher.py --moe -m /models/gguf/Hunyuan-A13B-Instruct-UD-Q6_K_XL-00001-of-00002.gguf
python llama-cpp-bencher.py --moe -m /models/gguf/dots.llm1.inst-UD-Q4_K_XL-00001-of-00002.gguf
