# run PETAL using WikiMIA

python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --length 32
python run.py --gpu_ids 0 --target_model falcon-7b --surrogate_model gpt2-xl --data WikiMIA --length 32
python run.py --gpu_ids 0 --target_model opt-6.7b --surrogate_model gpt2-xl --data WikiMIA --length 32
python run.py --gpu_ids 0 --target_model llama2-13b --surrogate_model gpt2-xl --data WikiMIA --length 32

# run PETAL using paraphrased WikiMIA

python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA-paraphrased --length 32
python run.py --gpu_ids 0 --target_model falcon-7b --surrogate_model gpt2-xl --data WikiMIA-paraphrased --length 32
python run.py --gpu_ids 0 --target_model opt-6.7b --surrogate_model gpt2-xl --data WikiMIA-paraphrased --length 32
python run.py --gpu_ids 0 --target_model llama2-13b --surrogate_model gpt2-xl --data WikiMIA-paraphrased --length 32

# run PETAL using MIMIR-DM Mathematics

python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data dm --length 32
python run.py --gpu_ids 0 --target_model pythia-2.8b --surrogate_model gpt2-xl --data dm --length 32
python run.py --gpu_ids 0 --target_model pythia-1.4b --surrogate_model gpt2-xl --data dm --length 32
python run.py --gpu_ids 0 --target_model pythia-160m --surrogate_model gpt2-xl --data dm --length 32

# run PETAL using Pythia-deduped

python run.py --gpu_ids 0 --target_model pythia-6.9b-dedup --surrogate_model gpt2-xl --data github --length 32
python run.py --gpu_ids 0 --target_model pythia-2.8b-dedup --surrogate_model gpt2-xl --data github --length 32
python run.py --gpu_ids 0 --target_model pythia-1.4b-dedup --surrogate_model gpt2-xl --data github --length 32
python run.py --gpu_ids 0 --target_model pythia-160m-dedup --surrogate_model gpt2-xl --data github --length 32

# run PETAL with different text length settings

python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --length 64
python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --length 128
python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --length 256

# run PETAL with different decoding strategy settings

python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --decoding greedy
python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --decoding nuclear
python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --decoding contrastive

# run PETAL using different sentence transformers

python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --embedding_model bge-large-en-v1.5
python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --embedding_model all-MiniLM-L6-v2
python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --embedding_model UAE-Large-V1
python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --embedding_model mxbai-embed-large-v1
