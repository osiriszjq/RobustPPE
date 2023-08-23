# models trained end-to-end
for model in 'pct' 'pctc' 'pctcmean' 'peat' 'peatmean' 'rscnn' 'pointnet2' 'dgcnn' 'pemax' 'pemean' 'pemedian' 'pointnetmean' 'pointnet' 'pointnetmlp3mean'; do

CUDA_VISIBLE_DEVICES=0 python main.py --exp-config configs/dgcnn_${model}_run_1.yaml

done

# analytical or random initialized per-point embedding
for model in 'pctc' 'pctcmean' 'peat' 'peatmean' 'pointnetmean' 'pointnet' 'pointnetmlp3mean'; do

CUDA_VISIBLE_DEVICES=1 python main.py --exp-config configs/dgcnn_${model}_run_1.yaml --random_init

done