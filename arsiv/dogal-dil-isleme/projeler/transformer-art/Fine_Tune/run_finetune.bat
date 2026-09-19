@echo off
echo Clearing CUDA cache and restarting fine-tuning...
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"
python fine_tune.py
