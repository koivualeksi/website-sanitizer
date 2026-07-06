======================================================================
Model:       unsloth/Qwen2.5-0.5B-Instruct
Phases:      ['sft', 'grpo']
Max seq len: 8192
Data dir:    /workspace/data
Output dir:  /workspace/output
======================================================================
Loaded 1774 train, 444 test samples
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2025.6.3: Fast Qwen2 patching. Transformers: 4.51.3.
   \\   /|    NVIDIA GeForce RTX 4090. Num GPUs = 1. Max memory: 23.527 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.0+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.3.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.30. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth 2025.6.3 patched 24 layers with 24 QKV layers, 24 O layers and 24 MLP layers.
trainable params: 8,798,208 || all params: 502,830,976 || trainable%: 1.7497
Building grammar FSM token masks...
  State 0: 10 valid tokens
  State 1: 11 valid tokens
  State 2: 10 valid tokens
  State 3: 12 valid tokens  [ACCEPT]
  State 4: 10 valid tokens
  State 5: 11 valid tokens
  State 6: 10 valid tokens
  State 7: 12 valid tokens  [ACCEPT]
  Walk '15:85': OK
  Walk '15:85,90:120': OK
  Walk '1:1': OK

======================================================================
PHASE: SFT (grammar-masked)
======================================================================
Tokenized: 1774 samples, 1631 with targets (143 dropped as truncated)

SFT: 1631 samples, 3 epochs, 611 steps, bf16=True
Unsloth: Will smartly offload gradients to save VRAM!
  step 10/611  loss=12.1181  lr=1.97e-04
  step 20/611  loss=7.6132  lr=1.93e-04
  step 30/611  loss=6.3740  lr=1.90e-04
  step 40/611  loss=4.8357  lr=1.87e-04
  step 50/611  loss=4.3031  lr=1.84e-04
  step 60/611  loss=3.5485  lr=1.80e-04
  step 70/611  loss=4.9754  lr=1.77e-04
  step 80/611  loss=4.1385  lr=1.74e-04
  step 90/611  loss=3.2339  lr=1.71e-04
  step 100/611  loss=3.6641  lr=1.67e-04
  step 110/611  loss=3.2600  lr=1.64e-04
  step 120/611  loss=3.3915  lr=1.61e-04
  step 130/611  loss=3.0899  lr=1.57e-04
  step 140/611  loss=3.6166  lr=1.54e-04
  step 150/611  loss=2.4658  lr=1.51e-04
  step 160/611  loss=3.4774  lr=1.48e-04
  step 170/611  loss=2.7390  lr=1.44e-04
  step 180/611  loss=2.6094  lr=1.41e-04
  step 190/611  loss=3.7925  lr=1.38e-04
  step 200/611  loss=3.0493  lr=1.35e-04
  Epoch 1/3 done.
  step 210/611  loss=2.8309  lr=1.31e-04
  step 220/611  loss=2.4856  lr=1.28e-04
  step 230/611  loss=1.0986  lr=1.25e-04
  step 240/611  loss=1.8396  lr=1.21e-04
  step 250/611  loss=2.1460  lr=1.18e-04
  step 260/611  loss=2.3950  lr=1.15e-04
  step 270/611  loss=1.4739  lr=1.12e-04
  step 280/611  loss=2.0978  lr=1.08e-04
  step 290/611  loss=2.5877  lr=1.05e-04
  step 300/611  loss=2.1066  lr=1.02e-04
  step 310/611  loss=1.4369  lr=9.85e-05
  step 320/611  loss=2.0320  lr=9.53e-05
  step 330/611  loss=2.5464  lr=9.20e-05
  step 340/611  loss=2.3457  lr=8.87e-05
  step 350/611  loss=2.1858  lr=8.54e-05
  step 360/611  loss=2.3007  lr=8.22e-05
  step 370/611  loss=1.7015  lr=7.89e-05
  step 380/611  loss=2.0448  lr=7.56e-05
  step 390/611  loss=2.0333  lr=7.23e-05
  step 400/611  loss=2.1901  lr=6.91e-05
  Epoch 2/3 done.
  step 410/611  loss=2.1218  lr=6.58e-05
  step 420/611  loss=1.2275  lr=6.25e-05
  step 430/611  loss=1.3247  lr=5.92e-05
  step 440/611  loss=1.0161  lr=5.60e-05
  step 450/611  loss=1.0317  lr=5.27e-05
  step 460/611  loss=1.1115  lr=4.94e-05
  step 470/611  loss=0.7769  lr=4.62e-05
  step 480/611  loss=1.7058  lr=4.29e-05
  step 490/611  loss=1.3037  lr=3.96e-05
  step 500/611  loss=0.9465  lr=3.63e-05
  step 510/611  loss=0.6874  lr=3.31e-05
  step 520/611  loss=0.7686  lr=2.98e-05
  step 530/611  loss=1.3774  lr=2.65e-05
  step 540/611  loss=1.0240  lr=2.32e-05
  step 550/611  loss=0.9760  lr=2.00e-05
  step 560/611  loss=1.2284  lr=1.67e-05
  step 570/611  loss=1.3280  lr=1.34e-05
  step 580/611  loss=1.1600  lr=1.01e-05
  step 590/611  loss=1.5549  lr=6.87e-06
  step 600/611  loss=1.0246  lr=3.60e-06
  Epoch 3/3 done.
SFT complete in 1257s
Saved SFT adapter to /workspace/output/sft-0.5b

--- SFT Eval (unconstrained) ---
  [2/444] pid=5 PARSE_ERROR: 22
  [5/444] pid=19 PARSE_ERROR: 56:79:80
  [100/444] running IoU=0.895
  [200/444] running IoU=0.896
  [300/444] running IoU=0.904
  [400/444] running IoU=0.907

SFT (free) — 367/444 succeeded (77 parse failures)
  Mean IoU:  0.903
  Mean Prec: 0.932
  Mean Rec:  0.958

--- SFT Eval (constrained) ---
  [100/444] running IoU=0.858
  [200/444] running IoU=0.870
  [300/444] running IoU=0.878
  [400/444] running IoU=0.874

SFT (constrained) — 413/444 succeeded (31 parse failures)
  Mean IoU:  0.867
  Mean Prec: 0.919
  Mean Rec:  0.923

======================================================================
PHASE: GRPO (constrained generation)
======================================================================

GRPO: 1622 samples (152 skipped as too long)
Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.
We will change the batch size of 1 to the `num_generations` of 4
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 1,622 | Num Epochs = 1 | Total steps = 405
O^O/ \_/ \    Batch size per device = 4 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (4 x 4 x 1) = 16
 "-____-"     Trainable parameters = 8,798,208/500,000,000 (1.76% trained)

  0%|          | 0/405 [00:00<?, ?it/s]
  0%|          | 1/405 [00:22<2:31:38, 22.52s/it]
  0%|          | 2/405 [00:45<2:32:28, 22.70s/it]
  1%|          | 3/405 [01:09<2:35:46, 23.25s/it]
  1%|          | 4/405 [01:31<2:33:18, 22.94s/it]
  1%|          | 5/405 [01:53<2:29:33, 22.43s/it]
  1%|▏         | 6/405 [02:14<2:27:13, 22.14s/it]
  2%|▏         | 7/405 [02:37<2:28:01, 22.32s/it]
  2%|▏         | 8/405 [03:00<2:29:29, 22.59s/it]
  2%|▏         | 9/405 [03:22<2:27:50, 22.40s/it]
  2%|▏         | 10/405 [03:44<2:26:29, 22.25s/it]
                                                  
{'loss': 1083.2921, 'grad_norm': 421403.8125, 'learning_rate': 1.0975609756097562e-06, 'num_tokens': 332888.0, 'completions/mean_length': 124.95, 'completions/min_length': 124.95, 'completions/max_length': 124.95, 'completions/clipped_ratio': 0.975, 'completions/mean_terminated_length': 0.15, 'completions/min_terminated_length': 0.15, 'completions/max_terminated_length': 0.15, 'rewards/fbeta_reward/mean': 0.024219439923763277, 'rewards/fbeta_reward/std': 0.001507145632058382, 'reward': 0.024219439923763277, 'reward_std': 0.0015071455389261245, 'frac_reward_zero_std': 0.975, 'completion_length': 124.95, 'kl': 10832.919916534423, 'epoch': 0.02}

  2%|▏         | 10/405 [03:44<2:26:29, 22.25s/it]
  3%|▎         | 11/405 [04:08<2:28:33, 22.62s/it]
  3%|▎         | 12/405 [04:29<2:26:34, 22.38s/it]
  3%|▎         | 13/405 [04:53<2:28:07, 22.67s/it]
  3%|▎         | 14/405 [05:17<2:30:26, 23.08s/it]
  4%|▎         | 15/405 [05:39<2:28:55, 22.91s/it]
  4%|▍         | 16/405 [06:02<2:27:41, 22.78s/it]
  4%|▍         | 17/405 [06:24<2:26:57, 22.72s/it]
  4%|▍         | 18/405 [06:46<2:25:27, 22.55s/it]
  5%|▍         | 19/405 [07:09<2:24:19, 22.43s/it]
  5%|▍         | 20/405 [07:33<2:26:54, 22.90s/it]
                                                  
{'loss': 167.8423, 'grad_norm': 100282.125, 'learning_rate': 2.317073170731708e-06, 'num_tokens': 768296.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 1678.4227701067925, 'epoch': 0.05}

  5%|▍         | 20/405 [07:33<2:26:54, 22.90s/it]
  5%|▌         | 21/405 [07:55<2:24:35, 22.59s/it]
  5%|▌         | 22/405 [08:16<2:22:11, 22.27s/it]
  6%|▌         | 23/405 [08:38<2:22:01, 22.31s/it]
  6%|▌         | 24/405 [09:01<2:21:22, 22.26s/it]
  6%|▌         | 25/405 [09:24<2:23:35, 22.67s/it]
  6%|▋         | 26/405 [09:46<2:22:06, 22.50s/it]
  7%|▋         | 27/405 [10:10<2:23:30, 22.78s/it]
  7%|▋         | 28/405 [10:34<2:25:56, 23.23s/it]
  7%|▋         | 29/405 [10:57<2:24:58, 23.13s/it]
  7%|▋         | 30/405 [11:20<2:25:07, 23.22s/it]
                                                  
{'loss': 90.6554, 'grad_norm': 10606.4384765625, 'learning_rate': 3.5365853658536588e-06, 'num_tokens': 1191152.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 906.5543153122068, 'epoch': 0.07}

  7%|▋         | 30/405 [11:20<2:25:07, 23.22s/it]
  8%|▊         | 31/405 [11:44<2:24:50, 23.24s/it]
  8%|▊         | 32/405 [12:08<2:26:24, 23.55s/it]
  8%|▊         | 33/405 [12:30<2:23:47, 23.19s/it]
  8%|▊         | 34/405 [12:52<2:20:52, 22.78s/it]
  9%|▊         | 35/405 [13:15<2:20:51, 22.84s/it]
  9%|▉         | 36/405 [13:39<2:22:50, 23.23s/it]
  9%|▉         | 37/405 [14:01<2:19:57, 22.82s/it]
  9%|▉         | 38/405 [14:24<2:20:37, 22.99s/it]
 10%|▉         | 39/405 [14:47<2:18:41, 22.74s/it]
 10%|▉         | 40/405 [15:09<2:17:30, 22.60s/it]
                                                  
{'loss': 129.1284, 'grad_norm': 34915.11328125, 'learning_rate': 4.75609756097561e-06, 'num_tokens': 1648856.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 1291.2842331424356, 'epoch': 0.1}

 10%|▉         | 40/405 [15:09<2:17:30, 22.60s/it]
 10%|█         | 41/405 [15:31<2:16:39, 22.53s/it]
 10%|█         | 42/405 [15:53<2:14:22, 22.21s/it]
 11%|█         | 43/405 [16:15<2:14:08, 22.23s/it]
 11%|█         | 44/405 [16:37<2:12:56, 22.10s/it]
 11%|█         | 45/405 [17:00<2:14:37, 22.44s/it]
 11%|█▏        | 46/405 [17:23<2:15:15, 22.61s/it]
 12%|█▏        | 47/405 [17:47<2:17:24, 23.03s/it]
 12%|█▏        | 48/405 [18:09<2:15:40, 22.80s/it]
 12%|█▏        | 49/405 [18:32<2:14:38, 22.69s/it]
 12%|█▏        | 50/405 [18:55<2:16:09, 23.01s/it]
                                                  
{'loss': 6.4956, 'grad_norm': 708.5657958984375, 'learning_rate': 4.890109890109891e-06, 'num_tokens': 2061700.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 64.95586165338754, 'epoch': 0.12}

 12%|█▏        | 50/405 [18:55<2:16:09, 23.01s/it]
 13%|█▎        | 51/405 [19:18<2:14:45, 22.84s/it]
 13%|█▎        | 52/405 [19:41<2:14:45, 22.90s/it]
 13%|█▎        | 53/405 [20:03<2:13:04, 22.68s/it]
 13%|█▎        | 54/405 [20:26<2:12:12, 22.60s/it]
 14%|█▎        | 55/405 [20:47<2:09:35, 22.22s/it]
 14%|█▍        | 56/405 [21:11<2:12:41, 22.81s/it]
 14%|█▍        | 57/405 [21:34<2:12:23, 22.83s/it]
 14%|█▍        | 58/405 [21:56<2:10:24, 22.55s/it]
 15%|█▍        | 59/405 [22:18<2:09:39, 22.48s/it]
 15%|█▍        | 60/405 [22:41<2:10:42, 22.73s/it]
                                                  
{'loss': 1.3984, 'grad_norm': 59.532508850097656, 'learning_rate': 4.752747252747253e-06, 'num_tokens': 2509784.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 13.983740267530084, 'epoch': 0.15}

 15%|█▍        | 60/405 [22:41<2:10:42, 22.73s/it]
 15%|█▌        | 61/405 [23:03<2:09:05, 22.52s/it]
 15%|█▌        | 62/405 [23:27<2:10:39, 22.86s/it]
 16%|█▌        | 63/405 [23:49<2:08:40, 22.57s/it]
 16%|█▌        | 64/405 [24:12<2:09:18, 22.75s/it]
 16%|█▌        | 65/405 [24:35<2:09:07, 22.79s/it]
 16%|█▋        | 66/405 [24:57<2:06:54, 22.46s/it]
 17%|█▋        | 67/405 [25:19<2:06:03, 22.38s/it]
 17%|█▋        | 68/405 [25:43<2:07:48, 22.75s/it]
 17%|█▋        | 69/405 [26:07<2:10:47, 23.35s/it]
 17%|█▋        | 70/405 [26:30<2:08:37, 23.04s/it]
                                                  
{'loss': 0.5117, 'grad_norm': 517.3652954101562, 'learning_rate': 4.615384615384616e-06, 'num_tokens': 2940684.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 5.117370004858822, 'epoch': 0.17}

 17%|█▋        | 70/405 [26:30<2:08:37, 23.04s/it]
 18%|█▊        | 71/405 [26:52<2:06:48, 22.78s/it]
 18%|█▊        | 72/405 [27:16<2:09:05, 23.26s/it]
 18%|█▊        | 73/405 [27:40<2:08:47, 23.28s/it]
 18%|█▊        | 74/405 [28:01<2:05:09, 22.69s/it]
 19%|█▊        | 75/405 [28:25<2:06:36, 23.02s/it]
 19%|█▉        | 76/405 [28:49<2:08:22, 23.41s/it]
 19%|█▉        | 77/405 [29:11<2:06:23, 23.12s/it]
 19%|█▉        | 78/405 [29:33<2:03:34, 22.67s/it]
 20%|█▉        | 79/405 [29:54<2:01:12, 22.31s/it]
 20%|█▉        | 80/405 [30:20<2:05:37, 23.19s/it]
                                                  
{'loss': 0.2528, 'grad_norm': 1.130536675453186, 'learning_rate': 4.478021978021979e-06, 'num_tokens': 3409920.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 2.5276036127936097, 'epoch': 0.2}

 20%|█▉        | 80/405 [30:20<2:05:37, 23.19s/it]
 20%|██        | 81/405 [30:43<2:04:40, 23.09s/it]
 20%|██        | 82/405 [31:06<2:05:09, 23.25s/it]
 20%|██        | 83/405 [31:28<2:01:55, 22.72s/it]
 21%|██        | 84/405 [31:50<2:00:57, 22.61s/it]
 21%|██        | 85/405 [32:14<2:03:13, 23.10s/it]
 21%|██        | 86/405 [32:37<2:02:44, 23.09s/it]
 21%|██▏       | 87/405 [33:01<2:02:46, 23.16s/it]
 22%|██▏       | 88/405 [33:22<1:59:56, 22.70s/it]
 22%|██▏       | 89/405 [33:45<1:59:16, 22.65s/it]
 22%|██▏       | 90/405 [34:08<1:59:08, 22.69s/it]
                                                  
{'loss': 0.1064, 'grad_norm': 3.449852466583252, 'learning_rate': 4.340659340659341e-06, 'num_tokens': 3823160.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 1.0644793222658335, 'epoch': 0.22}

 22%|██▏       | 90/405 [34:08<1:59:08, 22.69s/it]
 22%|██▏       | 91/405 [34:31<1:59:56, 22.92s/it]
 23%|██▎       | 92/405 [34:55<2:01:16, 23.25s/it]
 23%|██▎       | 93/405 [35:18<1:59:54, 23.06s/it]
 23%|██▎       | 94/405 [35:41<1:59:21, 23.03s/it]
 23%|██▎       | 95/405 [36:03<1:58:31, 22.94s/it]
 24%|██▎       | 96/405 [36:26<1:58:21, 22.98s/it]
 24%|██▍       | 97/405 [36:50<1:58:14, 23.03s/it]
 24%|██▍       | 98/405 [37:14<1:59:28, 23.35s/it]
 24%|██▍       | 99/405 [37:37<1:58:59, 23.33s/it]
 25%|██▍       | 100/405 [37:59<1:56:55, 23.00s/it]
                                                   
{'loss': 0.0463, 'grad_norm': 5.825719833374023, 'learning_rate': 4.203296703296703e-06, 'num_tokens': 4266032.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.4625725506106392, 'epoch': 0.25}

 25%|██▍       | 100/405 [37:59<1:56:55, 23.00s/it]
 25%|██▍       | 101/405 [38:22<1:55:47, 22.85s/it]
 25%|██▌       | 102/405 [38:45<1:55:34, 22.89s/it]
 25%|██▌       | 103/405 [39:08<1:56:27, 23.14s/it]
 26%|██▌       | 104/405 [39:31<1:55:56, 23.11s/it]
 26%|██▌       | 105/405 [39:53<1:53:25, 22.69s/it]
 26%|██▌       | 106/405 [40:18<1:56:05, 23.29s/it]
 26%|██▋       | 107/405 [40:42<1:56:19, 23.42s/it]
 27%|██▋       | 108/405 [41:05<1:55:35, 23.35s/it]
 27%|██▋       | 109/405 [41:27<1:53:51, 23.08s/it]
 27%|██▋       | 110/405 [41:50<1:52:41, 22.92s/it]
                                                   
{'loss': 0.0323, 'grad_norm': 5.683651447296143, 'learning_rate': 4.065934065934066e-06, 'num_tokens': 4732080.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.3226355278864503, 'epoch': 0.27}

 27%|██▋       | 110/405 [41:50<1:52:41, 22.92s/it]
 27%|██▋       | 111/405 [42:11<1:50:10, 22.48s/it]
 28%|██▊       | 112/405 [42:33<1:48:41, 22.26s/it]
 28%|██▊       | 113/405 [42:56<1:49:14, 22.45s/it]
 28%|██▊       | 114/405 [43:18<1:49:04, 22.49s/it]
 28%|██▊       | 115/405 [43:41<1:48:08, 22.37s/it]
 29%|██▊       | 116/405 [44:03<1:48:32, 22.53s/it]
 29%|██▉       | 117/405 [44:26<1:47:45, 22.45s/it]
 29%|██▉       | 118/405 [44:48<1:46:48, 22.33s/it]
 29%|██▉       | 119/405 [45:10<1:46:28, 22.34s/it]
 30%|██▉       | 120/405 [45:32<1:45:22, 22.18s/it]
                                                   
{'loss': 0.0197, 'grad_norm': 0.09220623970031738, 'learning_rate': 3.928571428571429e-06, 'num_tokens': 5072388.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.19667731868103147, 'epoch': 0.3}

 30%|██▉       | 120/405 [45:32<1:45:22, 22.18s/it]
 30%|██▉       | 121/405 [45:56<1:48:10, 22.85s/it]
 30%|███       | 122/405 [46:20<1:49:28, 23.21s/it]
 30%|███       | 123/405 [46:43<1:48:24, 23.07s/it]
 31%|███       | 124/405 [47:06<1:47:03, 22.86s/it]
 31%|███       | 125/405 [47:29<1:46:55, 22.91s/it]
 31%|███       | 126/405 [47:50<1:45:03, 22.59s/it]
 31%|███▏      | 127/405 [48:12<1:43:37, 22.36s/it]
 32%|███▏      | 128/405 [48:37<1:46:14, 23.01s/it]
 32%|███▏      | 129/405 [48:59<1:44:19, 22.68s/it]
 32%|███▏      | 130/405 [49:22<1:44:47, 22.86s/it]
                                                   
{'loss': 0.0076, 'grad_norm': 0.33097848296165466, 'learning_rate': 3.7912087912087915e-06, 'num_tokens': 5541600.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.07617088537663222, 'epoch': 0.32}

 32%|███▏      | 130/405 [49:22<1:44:47, 22.86s/it]
 32%|███▏      | 131/405 [49:44<1:42:46, 22.50s/it]
 33%|███▎      | 132/405 [50:06<1:42:41, 22.57s/it]
 33%|███▎      | 133/405 [50:29<1:41:45, 22.45s/it]
 33%|███▎      | 134/405 [50:50<1:40:10, 22.18s/it]
 33%|███▎      | 135/405 [51:12<1:39:30, 22.11s/it]
 34%|███▎      | 136/405 [51:34<1:38:36, 21.99s/it]
 34%|███▍      | 137/405 [51:55<1:37:46, 21.89s/it]
 34%|███▍      | 138/405 [52:17<1:37:33, 21.92s/it]
 34%|███▍      | 139/405 [52:40<1:37:27, 21.98s/it]
 35%|███▍      | 140/405 [53:02<1:37:14, 22.02s/it]
                                                   
{'loss': 0.0072, 'grad_norm': 0.10964173078536987, 'learning_rate': 3.653846153846154e-06, 'num_tokens': 5860216.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.0718333023134619, 'epoch': 0.35}

 35%|███▍      | 140/405 [53:02<1:37:14, 22.02s/it]
 35%|███▍      | 141/405 [53:24<1:37:53, 22.25s/it]
 35%|███▌      | 142/405 [53:46<1:37:19, 22.20s/it]
 35%|███▌      | 143/405 [54:08<1:36:17, 22.05s/it]
 36%|███▌      | 144/405 [54:30<1:35:58, 22.06s/it]
 36%|███▌      | 145/405 [54:52<1:34:38, 21.84s/it]
 36%|███▌      | 146/405 [55:13<1:34:11, 21.82s/it]
 36%|███▋      | 147/405 [55:36<1:35:07, 22.12s/it]
 37%|███▋      | 148/405 [55:59<1:35:13, 22.23s/it]
 37%|███▋      | 149/405 [56:21<1:34:35, 22.17s/it]
 37%|███▋      | 150/405 [56:43<1:33:55, 22.10s/it]
                                                   
{'loss': 0.0043, 'grad_norm': 0.06204722449183464, 'learning_rate': 3.516483516483517e-06, 'num_tokens': 6209112.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.042894990323111414, 'epoch': 0.37}

 37%|███▋      | 150/405 [56:43<1:33:55, 22.10s/it]
 37%|███▋      | 151/405 [57:06<1:34:37, 22.35s/it]
 38%|███▊      | 152/405 [57:28<1:33:49, 22.25s/it]
 38%|███▊      | 153/405 [57:50<1:33:39, 22.30s/it]
 38%|███▊      | 154/405 [58:12<1:33:03, 22.25s/it]
 38%|███▊      | 155/405 [58:34<1:31:50, 22.04s/it]
 39%|███▊      | 156/405 [58:57<1:33:18, 22.48s/it]
 39%|███▉      | 157/405 [59:20<1:33:12, 22.55s/it]
 39%|███▉      | 158/405 [59:43<1:33:30, 22.72s/it]
 39%|███▉      | 159/405 [1:00:06<1:33:04, 22.70s/it]
 40%|███▉      | 160/405 [1:00:29<1:34:01, 23.03s/it]
                                                     
{'loss': 0.0093, 'grad_norm': 0.4598151445388794, 'learning_rate': 3.3791208791208797e-06, 'num_tokens': 6690284.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.09339675505179912, 'epoch': 0.39}

 40%|███▉      | 160/405 [1:00:29<1:34:01, 23.03s/it]
 40%|███▉      | 161/405 [1:00:52<1:32:47, 22.82s/it]
 40%|████      | 162/405 [1:01:15<1:33:22, 23.05s/it]
 40%|████      | 163/405 [1:01:39<1:33:49, 23.26s/it]
 40%|████      | 164/405 [1:02:02<1:33:24, 23.26s/it]
 41%|████      | 165/405 [1:02:24<1:31:27, 22.87s/it]
 41%|████      | 166/405 [1:02:46<1:29:50, 22.55s/it]
 41%|████      | 167/405 [1:03:09<1:29:59, 22.69s/it]
 41%|████▏     | 168/405 [1:03:32<1:29:36, 22.68s/it]
 42%|████▏     | 169/405 [1:03:54<1:28:55, 22.61s/it]
 42%|████▏     | 170/405 [1:04:17<1:28:13, 22.53s/it]
                                                     
{'loss': 0.0113, 'grad_norm': 6.956180095672607, 'learning_rate': 3.2417582417582424e-06, 'num_tokens': 7087480.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.11275720925768837, 'epoch': 0.42}

 42%|████▏     | 170/405 [1:04:17<1:28:13, 22.53s/it]
 42%|████▏     | 171/405 [1:04:41<1:29:56, 23.06s/it]
 42%|████▏     | 172/405 [1:05:04<1:29:30, 23.05s/it]
 43%|████▎     | 173/405 [1:05:25<1:27:05, 22.52s/it]
 43%|████▎     | 174/405 [1:05:48<1:26:48, 22.55s/it]
 43%|████▎     | 175/405 [1:06:10<1:26:32, 22.57s/it]
 43%|████▎     | 176/405 [1:06:33<1:25:58, 22.53s/it]
 44%|████▎     | 177/405 [1:06:55<1:25:15, 22.43s/it]
 44%|████▍     | 178/405 [1:07:17<1:24:11, 22.25s/it]
 44%|████▍     | 179/405 [1:07:39<1:23:28, 22.16s/it]
 44%|████▍     | 180/405 [1:08:00<1:22:27, 21.99s/it]
                                                     
{'loss': 0.0052, 'grad_norm': 0.6036176681518555, 'learning_rate': 3.1043956043956046e-06, 'num_tokens': 7479636.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.05208676322363317, 'epoch': 0.44}

 44%|████▍     | 180/405 [1:08:00<1:22:27, 21.99s/it]
 45%|████▍     | 181/405 [1:08:22<1:21:23, 21.80s/it]
 45%|████▍     | 182/405 [1:08:45<1:22:09, 22.10s/it]
 45%|████▌     | 183/405 [1:09:07<1:22:08, 22.20s/it]
 45%|████▌     | 184/405 [1:09:29<1:21:00, 21.99s/it]
 46%|████▌     | 185/405 [1:09:51<1:21:18, 22.17s/it]
 46%|████▌     | 186/405 [1:10:15<1:22:52, 22.71s/it]
 46%|████▌     | 187/405 [1:10:38<1:23:02, 22.86s/it]
 46%|████▋     | 188/405 [1:11:00<1:21:30, 22.54s/it]
 47%|████▋     | 189/405 [1:11:24<1:22:16, 22.85s/it]
 47%|████▋     | 190/405 [1:11:46<1:21:21, 22.70s/it]
                                                     
{'loss': 0.008, 'grad_norm': 0.3464931845664978, 'learning_rate': 2.9670329670329673e-06, 'num_tokens': 7888260.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.07966132431756705, 'epoch': 0.47}

 47%|████▋     | 190/405 [1:11:46<1:21:21, 22.70s/it]
 47%|████▋     | 191/405 [1:12:10<1:22:25, 23.11s/it]
 47%|████▋     | 192/405 [1:12:32<1:20:36, 22.70s/it]
 48%|████▊     | 193/405 [1:12:54<1:20:01, 22.65s/it]
 48%|████▊     | 194/405 [1:13:17<1:19:52, 22.71s/it]
 48%|████▊     | 195/405 [1:13:41<1:20:08, 22.90s/it]
 48%|████▊     | 196/405 [1:14:03<1:18:44, 22.61s/it]
 49%|████▊     | 197/405 [1:14:25<1:18:16, 22.58s/it]
 49%|████▉     | 198/405 [1:14:47<1:17:02, 22.33s/it]
 49%|████▉     | 199/405 [1:15:13<1:20:26, 23.43s/it]
 49%|████▉     | 200/405 [1:15:36<1:20:16, 23.49s/it]
                                                     
{'loss': 0.0088, 'grad_norm': 0.0817265659570694, 'learning_rate': 2.82967032967033e-06, 'num_tokens': 8348476.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.08800370085518808, 'epoch': 0.49}

 49%|████▉     | 200/405 [1:15:36<1:20:16, 23.49s/it]
 50%|████▉     | 201/405 [1:16:00<1:19:40, 23.44s/it]
 50%|████▉     | 202/405 [1:16:23<1:18:41, 23.26s/it]
 50%|█████     | 203/405 [1:16:45<1:17:34, 23.04s/it]
 50%|█████     | 204/405 [1:17:08<1:17:09, 23.03s/it]
 51%|█████     | 205/405 [1:17:32<1:17:59, 23.40s/it]
 51%|█████     | 206/405 [1:17:56<1:17:53, 23.49s/it]
 51%|█████     | 207/405 [1:18:19<1:16:31, 23.19s/it]
 51%|█████▏    | 208/405 [1:18:42<1:16:47, 23.39s/it]
 52%|█████▏    | 209/405 [1:19:05<1:15:29, 23.11s/it]
 52%|█████▏    | 210/405 [1:19:28<1:14:45, 23.00s/it]
                                                     
{'loss': 0.0056, 'grad_norm': 0.7187785506248474, 'learning_rate': 2.6923076923076923e-06, 'num_tokens': 8743296.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.05637760201934725, 'epoch': 0.52}

 52%|█████▏    | 210/405 [1:19:28<1:14:45, 23.00s/it]
 52%|█████▏    | 211/405 [1:19:51<1:14:45, 23.12s/it]
 52%|█████▏    | 212/405 [1:20:14<1:14:28, 23.15s/it]
 53%|█████▎    | 213/405 [1:20:38<1:14:44, 23.36s/it]
 53%|█████▎    | 214/405 [1:21:01<1:13:49, 23.19s/it]
 53%|█████▎    | 215/405 [1:21:24<1:13:47, 23.30s/it]
 53%|█████▎    | 216/405 [1:21:48<1:13:52, 23.45s/it]
 54%|█████▎    | 217/405 [1:22:12<1:13:35, 23.49s/it]
 54%|█████▍    | 218/405 [1:22:34<1:12:00, 23.11s/it]
 54%|█████▍    | 219/405 [1:22:57<1:11:14, 22.98s/it]
 54%|█████▍    | 220/405 [1:23:19<1:10:25, 22.84s/it]
                                                     
{'loss': 0.0148, 'grad_norm': 3.0860326290130615, 'learning_rate': 2.554945054945055e-06, 'num_tokens': 9183760.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.14792814487591385, 'epoch': 0.54}

 54%|█████▍    | 220/405 [1:23:19<1:10:25, 22.84s/it]
 55%|█████▍    | 221/405 [1:23:42<1:10:03, 22.84s/it]
 55%|█████▍    | 222/405 [1:24:06<1:10:45, 23.20s/it]
 55%|█████▌    | 223/405 [1:24:29<1:10:16, 23.17s/it]
 55%|█████▌    | 224/405 [1:24:51<1:08:50, 22.82s/it]
 56%|█████▌    | 225/405 [1:25:14<1:08:26, 22.81s/it]
 56%|█████▌    | 226/405 [1:25:36<1:07:26, 22.61s/it]
 56%|█████▌    | 227/405 [1:25:59<1:07:24, 22.72s/it]
 56%|█████▋    | 228/405 [1:26:21<1:06:39, 22.59s/it]
 57%|█████▋    | 229/405 [1:26:44<1:06:30, 22.68s/it]
 57%|█████▋    | 230/405 [1:27:06<1:05:14, 22.37s/it]
                                                     
{'loss': 0.0043, 'grad_norm': 0.11274182051420212, 'learning_rate': 2.4175824175824177e-06, 'num_tokens': 9575880.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.04267231950070709, 'epoch': 0.57}

 57%|█████▋    | 230/405 [1:27:06<1:05:14, 22.37s/it]
 57%|█████▋    | 231/405 [1:27:28<1:04:34, 22.27s/it]
 57%|█████▋    | 232/405 [1:27:50<1:04:20, 22.32s/it]
 58%|█████▊    | 233/405 [1:28:12<1:03:34, 22.18s/it]
 58%|█████▊    | 234/405 [1:28:34<1:02:36, 21.97s/it]
 58%|█████▊    | 235/405 [1:28:57<1:02:54, 22.20s/it]
 58%|█████▊    | 236/405 [1:29:19<1:02:49, 22.31s/it]
 59%|█████▊    | 237/405 [1:29:42<1:03:18, 22.61s/it]
 59%|█████▉    | 238/405 [1:30:06<1:03:54, 22.96s/it]
 59%|█████▉    | 239/405 [1:30:28<1:02:20, 22.54s/it]
 59%|█████▉    | 240/405 [1:30:50<1:01:48, 22.48s/it]
                                                     
{'loss': 0.0034, 'grad_norm': 0.11171606928110123, 'learning_rate': 2.2802197802197804e-06, 'num_tokens': 9903996.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.03435588451102376, 'epoch': 0.59}

 59%|█████▉    | 240/405 [1:30:50<1:01:48, 22.48s/it]
 60%|█████▉    | 241/405 [1:31:12<1:01:06, 22.35s/it]
 60%|█████▉    | 242/405 [1:31:34<1:00:03, 22.11s/it]
 60%|██████    | 243/405 [1:31:56<1:00:11, 22.29s/it]
 60%|██████    | 244/405 [1:32:19<59:58, 22.35s/it]  
 60%|██████    | 245/405 [1:32:43<1:00:58, 22.87s/it]
 61%|██████    | 246/405 [1:33:06<1:00:36, 22.87s/it]
 61%|██████    | 247/405 [1:33:29<1:00:43, 23.06s/it]
 61%|██████    | 248/405 [1:33:52<1:00:24, 23.08s/it]
 61%|██████▏   | 249/405 [1:34:15<59:15, 22.79s/it]  
 62%|██████▏   | 250/405 [1:34:37<58:25, 22.61s/it]
                                                   
{'loss': 0.0045, 'grad_norm': 0.18180473148822784, 'learning_rate': 2.1428571428571427e-06, 'num_tokens': 10347112.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.044952957122586665, 'epoch': 0.62}

 62%|██████▏   | 250/405 [1:34:37<58:25, 22.61s/it]
 62%|██████▏   | 251/405 [1:35:01<59:06, 23.03s/it]
 62%|██████▏   | 252/405 [1:35:24<58:40, 23.01s/it]
 62%|██████▏   | 253/405 [1:35:47<58:49, 23.22s/it]
 63%|██████▎   | 254/405 [1:36:09<57:29, 22.85s/it]
 63%|██████▎   | 255/405 [1:36:31<56:29, 22.59s/it]
 63%|██████▎   | 256/405 [1:36:54<55:48, 22.48s/it]
 63%|██████▎   | 257/405 [1:37:16<55:25, 22.47s/it]
 64%|██████▎   | 258/405 [1:37:38<54:35, 22.28s/it]
 64%|██████▍   | 259/405 [1:38:01<54:27, 22.38s/it]
 64%|██████▍   | 260/405 [1:38:23<53:56, 22.32s/it]
                                                   
{'loss': 0.0069, 'grad_norm': 0.07162930816411972, 'learning_rate': 2.005494505494506e-06, 'num_tokens': 10714288.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.06906272429041564, 'epoch': 0.64}

 64%|██████▍   | 260/405 [1:38:23<53:56, 22.32s/it]
 64%|██████▍   | 261/405 [1:38:45<53:37, 22.34s/it]
 65%|██████▍   | 262/405 [1:39:09<54:05, 22.70s/it]
 65%|██████▍   | 263/405 [1:39:30<52:54, 22.35s/it]
 65%|██████▌   | 264/405 [1:39:53<52:45, 22.45s/it]
 65%|██████▌   | 265/405 [1:40:17<53:21, 22.87s/it]
 66%|██████▌   | 266/405 [1:40:38<52:14, 22.55s/it]
 66%|██████▌   | 267/405 [1:41:00<51:15, 22.29s/it]
 66%|██████▌   | 268/405 [1:41:23<51:18, 22.47s/it]
 66%|██████▋   | 269/405 [1:41:47<51:44, 22.83s/it]
 67%|██████▋   | 270/405 [1:42:10<51:20, 22.82s/it]
                                                   
{'loss': 0.0047, 'grad_norm': 0.32008230686187744, 'learning_rate': 1.8681318681318684e-06, 'num_tokens': 11158812.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.04724722296232357, 'epoch': 0.67}

 67%|██████▋   | 270/405 [1:42:10<51:20, 22.82s/it]
 67%|██████▋   | 271/405 [1:42:33<51:18, 22.98s/it]
 67%|██████▋   | 272/405 [1:42:56<50:53, 22.96s/it]
 67%|██████▋   | 273/405 [1:43:19<50:46, 23.08s/it]
 68%|██████▊   | 274/405 [1:43:41<49:37, 22.73s/it]
 68%|██████▊   | 275/405 [1:44:03<48:55, 22.58s/it]
 68%|██████▊   | 276/405 [1:44:26<48:45, 22.68s/it]
 68%|██████▊   | 277/405 [1:44:50<49:19, 23.12s/it]
 69%|██████▊   | 278/405 [1:45:14<49:01, 23.16s/it]
 69%|██████▉   | 279/405 [1:45:36<47:59, 22.85s/it]
 69%|██████▉   | 280/405 [1:45:59<47:52, 22.98s/it]
                                                   
{'loss': 0.0057, 'grad_norm': 1.6247763633728027, 'learning_rate': 1.7307692307692308e-06, 'num_tokens': 11573512.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.05681213164934888, 'epoch': 0.69}

 69%|██████▉   | 280/405 [1:45:59<47:52, 22.98s/it]
 69%|██████▉   | 281/405 [1:46:21<47:08, 22.81s/it]
 70%|██████▉   | 282/405 [1:46:45<47:20, 23.10s/it]
 70%|██████▉   | 283/405 [1:47:08<46:31, 22.88s/it]
 70%|███████   | 284/405 [1:47:30<45:59, 22.80s/it]
 70%|███████   | 285/405 [1:47:55<46:33, 23.28s/it]
 71%|███████   | 286/405 [1:48:19<46:45, 23.58s/it]
 71%|███████   | 287/405 [1:48:41<45:33, 23.16s/it]
 71%|███████   | 288/405 [1:49:03<44:39, 22.90s/it]
 71%|███████▏  | 289/405 [1:49:26<43:59, 22.76s/it]
 72%|███████▏  | 290/405 [1:49:48<43:08, 22.51s/it]
                                                   
{'loss': 0.0035, 'grad_norm': 0.22265778481960297, 'learning_rate': 1.5934065934065933e-06, 'num_tokens': 11954940.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.03531516906805336, 'epoch': 0.72}

 72%|███████▏  | 290/405 [1:49:48<43:08, 22.51s/it]
 72%|███████▏  | 291/405 [1:50:12<43:40, 22.99s/it]
 72%|███████▏  | 292/405 [1:50:34<42:46, 22.71s/it]
 72%|███████▏  | 293/405 [1:50:55<41:41, 22.34s/it]
 73%|███████▎  | 294/405 [1:51:17<40:59, 22.16s/it]
 73%|███████▎  | 295/405 [1:51:40<41:17, 22.52s/it]
 73%|███████▎  | 296/405 [1:52:04<41:16, 22.72s/it]
 73%|███████▎  | 297/405 [1:52:27<41:26, 23.02s/it]
 74%|███████▎  | 298/405 [1:52:51<41:13, 23.11s/it]
 74%|███████▍  | 299/405 [1:53:13<40:32, 22.95s/it]
 74%|███████▍  | 300/405 [1:53:36<40:01, 22.88s/it]
                                                   
{'loss': 0.0043, 'grad_norm': 0.09818285703659058, 'learning_rate': 1.4560439560439563e-06, 'num_tokens': 12378412.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.04284783871844411, 'epoch': 0.74}

 74%|███████▍  | 300/405 [1:53:36<40:01, 22.88s/it]
 74%|███████▍  | 301/405 [1:53:59<39:43, 22.92s/it]
 75%|███████▍  | 302/405 [1:54:22<39:30, 23.02s/it]
 75%|███████▍  | 303/405 [1:54:46<39:16, 23.11s/it]
 75%|███████▌  | 304/405 [1:55:09<39:14, 23.31s/it]
 75%|███████▌  | 305/405 [1:55:34<39:47, 23.87s/it]
 76%|███████▌  | 306/405 [1:55:59<39:27, 23.91s/it]
 76%|███████▌  | 307/405 [1:56:21<38:12, 23.40s/it]
 76%|███████▌  | 308/405 [1:56:43<37:18, 23.08s/it]
 76%|███████▋  | 309/405 [1:57:06<37:03, 23.16s/it]
 77%|███████▋  | 310/405 [1:57:30<36:39, 23.16s/it]
                                                   
{'loss': 0.0045, 'grad_norm': 0.09518235176801682, 'learning_rate': 1.3186813186813187e-06, 'num_tokens': 12868284.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.044676373316906395, 'epoch': 0.76}

 77%|███████▋  | 310/405 [1:57:30<36:39, 23.16s/it]
 77%|███████▋  | 311/405 [1:57:53<36:30, 23.31s/it]
 77%|███████▋  | 312/405 [1:58:16<35:46, 23.09s/it]
 77%|███████▋  | 313/405 [1:58:38<35:10, 22.94s/it]
 78%|███████▊  | 314/405 [1:59:00<34:23, 22.67s/it]
 78%|███████▊  | 315/405 [1:59:23<34:07, 22.75s/it]
 78%|███████▊  | 316/405 [1:59:49<34:49, 23.47s/it]
 78%|███████▊  | 317/405 [2:00:11<34:11, 23.31s/it]
 79%|███████▊  | 318/405 [2:00:34<33:19, 22.98s/it]
 79%|███████▉  | 319/405 [2:00:57<33:05, 23.09s/it]
 79%|███████▉  | 320/405 [2:01:20<32:38, 23.05s/it]
                                                   
{'loss': 0.0038, 'grad_norm': 0.0829634964466095, 'learning_rate': 1.1813186813186815e-06, 'num_tokens': 13286336.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.03804305247031152, 'epoch': 0.79}

 79%|███████▉  | 320/405 [2:01:20<32:38, 23.05s/it]
 79%|███████▉  | 321/405 [2:01:43<32:21, 23.11s/it]
 80%|███████▉  | 322/405 [2:02:05<31:22, 22.69s/it]
 80%|███████▉  | 323/405 [2:02:26<30:31, 22.34s/it]
 80%|████████  | 324/405 [2:02:49<30:14, 22.40s/it]
 80%|████████  | 325/405 [2:03:11<29:38, 22.23s/it]
 80%|████████  | 326/405 [2:03:35<29:53, 22.70s/it]
 81%|████████  | 327/405 [2:03:59<30:20, 23.34s/it]
 81%|████████  | 328/405 [2:04:22<29:47, 23.22s/it]
 81%|████████  | 329/405 [2:04:46<29:26, 23.24s/it]
 81%|████████▏ | 330/405 [2:05:09<29:16, 23.42s/it]
                                                   
{'loss': 0.0041, 'grad_norm': 0.0983191654086113, 'learning_rate': 1.0439560439560442e-06, 'num_tokens': 13779504.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.040768169611692426, 'epoch': 0.81}

 81%|████████▏ | 330/405 [2:05:09<29:16, 23.42s/it]
 82%|████████▏ | 331/405 [2:05:34<29:19, 23.78s/it]
 82%|████████▏ | 332/405 [2:05:56<28:04, 23.07s/it]
 82%|████████▏ | 333/405 [2:06:18<27:24, 22.84s/it]
 82%|████████▏ | 334/405 [2:06:40<26:56, 22.76s/it]
 83%|████████▎ | 335/405 [2:07:03<26:36, 22.80s/it]
 83%|████████▎ | 336/405 [2:07:26<26:08, 22.74s/it]
 83%|████████▎ | 337/405 [2:07:49<25:53, 22.84s/it]
 83%|████████▎ | 338/405 [2:08:11<25:13, 22.59s/it]
 84%|████████▎ | 339/405 [2:08:33<24:45, 22.51s/it]
 84%|████████▍ | 340/405 [2:08:57<24:45, 22.85s/it]
                                                   
{'loss': 0.0048, 'grad_norm': 0.07922953367233276, 'learning_rate': 9.065934065934067e-07, 'num_tokens': 14239728.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.048464483220595864, 'epoch': 0.84}

 84%|████████▍ | 340/405 [2:08:57<24:45, 22.85s/it]
 84%|████████▍ | 341/405 [2:09:20<24:29, 22.97s/it]
 84%|████████▍ | 342/405 [2:09:42<23:40, 22.54s/it]
 85%|████████▍ | 343/405 [2:10:06<23:40, 22.92s/it]
 85%|████████▍ | 344/405 [2:10:27<22:52, 22.49s/it]
 85%|████████▌ | 345/405 [2:10:51<22:51, 22.86s/it]
 85%|████████▌ | 346/405 [2:11:14<22:35, 22.97s/it]
 86%|████████▌ | 347/405 [2:11:37<22:16, 23.04s/it]
 86%|████████▌ | 348/405 [2:12:00<21:48, 22.96s/it]
 86%|████████▌ | 349/405 [2:12:23<21:19, 22.85s/it]
 86%|████████▋ | 350/405 [2:12:45<20:52, 22.76s/it]
                                                   
{'loss': 0.0042, 'grad_norm': 0.10644781589508057, 'learning_rate': 7.692307692307694e-07, 'num_tokens': 14683332.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.04227151218801737, 'epoch': 0.86}

 86%|████████▋ | 350/405 [2:12:45<20:52, 22.76s/it]
 87%|████████▋ | 351/405 [2:13:08<20:24, 22.68s/it]
 87%|████████▋ | 352/405 [2:13:30<19:56, 22.57s/it]
 87%|████████▋ | 353/405 [2:13:52<19:25, 22.42s/it]
 87%|████████▋ | 354/405 [2:14:14<18:57, 22.31s/it]
 88%|████████▊ | 355/405 [2:14:36<18:23, 22.06s/it]
 88%|████████▊ | 356/405 [2:14:57<17:58, 22.02s/it]
 88%|████████▊ | 357/405 [2:15:21<17:59, 22.48s/it]
 88%|████████▊ | 358/405 [2:15:43<17:27, 22.28s/it]
 89%|████████▊ | 359/405 [2:16:06<17:22, 22.67s/it]
 89%|████████▉ | 360/405 [2:16:29<16:56, 22.59s/it]
                                                   
{'loss': 0.0044, 'grad_norm': 0.10619528591632843, 'learning_rate': 6.318681318681319e-07, 'num_tokens': 15042748.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.043858430441468955, 'epoch': 0.89}

 89%|████████▉ | 360/405 [2:16:29<16:56, 22.59s/it]
 89%|████████▉ | 361/405 [2:16:52<16:37, 22.68s/it]
 89%|████████▉ | 362/405 [2:17:13<16:03, 22.40s/it]
 90%|████████▉ | 363/405 [2:17:36<15:38, 22.34s/it]
 90%|████████▉ | 364/405 [2:17:58<15:13, 22.28s/it]
 90%|█████████ | 365/405 [2:18:20<14:46, 22.16s/it]
 90%|█████████ | 366/405 [2:18:43<14:42, 22.63s/it]
 91%|█████████ | 367/405 [2:19:06<14:14, 22.48s/it]
 91%|█████████ | 368/405 [2:19:27<13:44, 22.29s/it]
 91%|█████████ | 369/405 [2:19:49<13:12, 22.01s/it]
 91%|█████████▏| 370/405 [2:20:10<12:44, 21.83s/it]
                                                   
{'loss': 0.004, 'grad_norm': 0.5061593651771545, 'learning_rate': 4.945054945054946e-07, 'num_tokens': 15383528.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.0404347216244787, 'epoch': 0.91}

 91%|█████████▏| 370/405 [2:20:10<12:44, 21.83s/it]
 92%|█████████▏| 371/405 [2:20:34<12:45, 22.51s/it]
 92%|█████████▏| 372/405 [2:20:58<12:32, 22.80s/it]
 92%|█████████▏| 373/405 [2:21:22<12:19, 23.11s/it]
 92%|█████████▏| 374/405 [2:21:44<11:47, 22.83s/it]
 93%|█████████▎| 375/405 [2:22:06<11:18, 22.61s/it]
 93%|█████████▎| 376/405 [2:22:27<10:46, 22.30s/it]
 93%|█████████▎| 377/405 [2:22:50<10:23, 22.27s/it]
 93%|█████████▎| 378/405 [2:23:11<09:54, 22.01s/it]
 94%|█████████▎| 379/405 [2:23:33<09:32, 22.02s/it]
 94%|█████████▍| 380/405 [2:23:55<09:08, 21.92s/it]
                                                   
{'loss': 0.0036, 'grad_norm': 0.08959510177373886, 'learning_rate': 3.5714285714285716e-07, 'num_tokens': 15770852.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.03584747770801187, 'epoch': 0.94}

 94%|█████████▍| 380/405 [2:23:55<09:08, 21.92s/it]
 94%|█████████▍| 381/405 [2:24:17<08:47, 21.99s/it]
 94%|█████████▍| 382/405 [2:24:39<08:27, 22.08s/it]
 95%|█████████▍| 383/405 [2:25:01<08:06, 22.10s/it]
 95%|█████████▍| 384/405 [2:25:23<07:40, 21.93s/it]
 95%|█████████▌| 385/405 [2:25:45<07:18, 21.95s/it]
 95%|█████████▌| 386/405 [2:26:07<06:59, 22.10s/it]
 96%|█████████▌| 387/405 [2:26:29<06:33, 21.88s/it]
 96%|█████████▌| 388/405 [2:26:52<06:17, 22.19s/it]
 96%|█████████▌| 389/405 [2:27:14<05:56, 22.25s/it]
 96%|█████████▋| 390/405 [2:27:37<05:38, 22.60s/it]
                                                   
{'loss': 0.0038, 'grad_norm': 0.18963196873664856, 'learning_rate': 2.197802197802198e-07, 'num_tokens': 16137848.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.037822312966454774, 'epoch': 0.96}

 96%|█████████▋| 390/405 [2:27:37<05:38, 22.60s/it]
 97%|█████████▋| 391/405 [2:27:59<05:13, 22.40s/it]
 97%|█████████▋| 392/405 [2:28:22<04:51, 22.41s/it]
 97%|█████████▋| 393/405 [2:28:46<04:34, 22.84s/it]
 97%|█████████▋| 394/405 [2:29:07<04:08, 22.55s/it]
 98%|█████████▊| 395/405 [2:29:30<03:44, 22.42s/it]
 98%|█████████▊| 396/405 [2:29:52<03:22, 22.50s/it]
 98%|█████████▊| 397/405 [2:30:14<02:59, 22.41s/it]
 98%|█████████▊| 398/405 [2:30:36<02:35, 22.15s/it]
 99%|█████████▊| 399/405 [2:31:00<02:16, 22.67s/it]
 99%|█████████▉| 400/405 [2:31:23<01:53, 22.74s/it]
                                                   
{'loss': 0.0038, 'grad_norm': 0.30719494819641113, 'learning_rate': 8.241758241758242e-08, 'num_tokens': 16540952.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.03847967041656375, 'epoch': 0.99}

 99%|█████████▉| 400/405 [2:31:23<01:53, 22.74s/it]
 99%|█████████▉| 401/405 [2:31:48<01:34, 23.53s/it]
 99%|█████████▉| 402/405 [2:32:10<01:09, 23.08s/it]
100%|█████████▉| 403/405 [2:32:34<00:46, 23.27s/it]
100%|█████████▉| 404/405 [2:32:57<00:23, 23.14s/it]
100%|██████████| 405/405 [2:33:19<00:00, 22.90s/it]
                                                   
{'train_runtime': 9199.5965, 'train_samples_per_second': 0.176, 'train_steps_per_second': 0.044, 'train_loss': 36.54181984199878, 'num_tokens': 16778780.0, 'completions/mean_length': 128.0, 'completions/min_length': 128.0, 'completions/max_length': 128.0, 'completions/clipped_ratio': 1.0, 'completions/mean_terminated_length': 0.0, 'completions/min_terminated_length': 0.0, 'completions/max_terminated_length': 0.0, 'rewards/fbeta_reward/mean': 0.0, 'rewards/fbeta_reward/std': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'completion_length': 128.0, 'kl': 0.03723125127144158, 'epoch': 1.0}

100%|██████████| 405/405 [2:33:19<00:00, 22.90s/it]
100%|██████████| 405/405 [2:33:19<00:00, 22.72s/it]
GRPO time: 9200s
GRPO complete in 9201s
Saved GRPO adapter to /workspace/output/grpo-0.5b
==((====))==  Unsloth 2025.6.3: Fast Qwen2 patching. Transformers: 4.51.3.
   \\   /|    NVIDIA GeForce RTX 4090. Num GPUs = 1. Max memory: 23.527 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.0+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.3.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.30. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!

--- GRPO Eval (unconstrained) ---
  [2/444] pid=5 PARSE_ERROR: 23:24:25:26:27:28:29:30:31:32:33:34:35:36:37:38:39:40:41:42:43:44:45:46:47:48:49
  [3/444] pid=8 PARSE_ERROR: 64:66,67:60,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84
  [4/444] pid=15 PARSE_ERROR: 30:77:78:79:80:81:82:83:84:85:86:87:88:89:90:91:92:93:94:95:96:97:98:99:100:101:
  [6/444] pid=22 PARSE_ERROR: 53 | 79 | 80 | 83 | 96 | 97 | 98 | 99 | 100 | 100 | 100 | 100 | 100 | 100 | 100 
  [7/444] pid=29 PARSE_ERROR: 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 
  [8/444] pid=32 PARSE_ERROR: 398 | 399 | 400 | 401 | 402 | 403 | 404 | 405 | 406 | 407 | 408 | 409 | 410 | 41
  [9/444] pid=33 PARSE_ERROR: 17:34:35:36:37:38:39:40:41:42:43:44:45:46:47:48:50:52:53:54:55:56:57:58:59:60:61
  [10/444] pid=50 PARSE_ERROR: 10:13
10:13
  [12/444] pid=83 PARSE_ERROR: 44:95:96:97:98:99:100:101:102:103:104:105:106:107:108:109:110:111:112:113:114:11
  [13/444] pid=85 PARSE_ERROR: 40:52:53:54:55:56:57:58:59:60:61:62:63:64:65:66:67:68:71:72:73:74:75:76:77:78:79
  [14/444] pid=90 PARSE_ERROR: 46 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 
  [15/444] pid=91 PARSE_ERROR: 118:160:160:160:160:160:160:160:160:160:160:160:160:160:160:160:160:160:160:160:
  [17/444] pid=107 PARSE_ERROR: 43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43:43
  [18/444] pid=108 PARSE_ERROR: 535 | 536 | 537 | 538 | 539 | 540 | 541 | 542 | 543 | 544 | 561 | 562 | 563 | 56
  [20/444] pid=113 PARSE_ERROR: 15-29
  [100/444] running IoU=0.893
  [200/444] running IoU=0.839
  [300/444] running IoU=0.879
  [400/444] running IoU=0.893
GRPO (free) — 105/444 succeeded (339 parse failures)
  Mean IoU:  0.884
  Mean Prec: 0.895
  Mean Rec:  0.960

--- GRPO Eval (constrained) ---
  [2/444] pid=5 PARSE_ERROR: 23:24,25:26,27:28,29:21,22:23,24:25,26:27,28:29,30:31,32:33,34:35,36:37,38:39,40
  [3/444] pid=8 PARSE_ERROR: 64:66,67:60,62:63,62:63,62:63,62:63,62:63,62:63,62:63,62:63,62:63,62:63,62:63,62
  [4/444] pid=15 PARSE_ERROR: 30:77,78:74,75:77,78:77,78:77,78:77,78:77,78:77,78:77,78:77,78:77,78:77,78:77,78
  [13/444] pid=85 PARSE_ERROR: 40:52,53:52,53:52,53:52,53:52,53:52,53:52,53:52,53:52,53:52,53:52,53:52,53:52,53
  [100/444] running IoU=0.827
  [200/444] running IoU=0.823
  [300/444] running IoU=0.840
  [400/444] running IoU=0.839

GRPO (constrained) — 352/444 succeeded (92 parse failures)
  Mean IoU:  0.833
  Mean Prec: 0.852
  Mean Rec:  0.905

Saved eval results to /workspace/output/eval_0.5b.json

======================================================================
DONE
======================================================================