Place the pre-trained checkpoint file here:
  mf_mska_best.keras  (~6 MB, float32)

Download from:
  https://github.com/oussama123-ai/MF_MSKA_Net/releases

Usage:
  python scripts/evaluate.py \
      --data_dir  ./data/brisc2025/segmentation_task/train \
      --ckpt_path ./weights/mf_mska_best.keras \
      --output_dir ./eval_results
