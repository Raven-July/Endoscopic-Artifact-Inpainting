python process.py \
    --input /root/autodl-tmp/datasets/Original/Teeth/images/ \
    --output /root/autodl-tmp/datasets/K-samples/Teeth/ \
    --sd_path /root/autodl-tmp/pretrained_models/yoso-delight-v0-4-base \
    --duck_path /root/autodl-tmp/pretrained_models/K-Samples/Duck16mod_Spec_99_K1.pt \
    --lama_dilate_size 9 \
    --mix_thresh 150 \
    # --disable_sd True \
    # --trad_help True \
    # sd_path路径需要在下载stabledelight模型后，根据实际位置做修改
    # teeth用9，其他用15(12)
        # --input /root/autodl-tmp/datasets/Original/company/20250217_104106_emUpperJaw/ \
