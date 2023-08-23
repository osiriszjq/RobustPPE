# evaluation on the our modified ModelNet40-C dataset


# models trained end-to-end
for iid in 1 2 3 4 5; do

if [ ! -d "modelnetc_our_"${iid} ]; then
    mkdir "modelnetc_our_"${iid}
fi

for model in  'pct' 'pctc' 'pctcmean' 'peat' 'peatmean' 'rscnn' 'pointnet2' 'dgcnn' 'pemax' 'pemean' 'pemedian' 'pointnetmean' 'pointnet' 'pointnetmlp3mean'; do 
for cor in 'gaussian' 'uniform' 'ball_l' 'ball_m' 'ball_h' 'background' 'impulse' 'upsampling'; do #
for sev in 1 2 3 4 5 6 7 8 9 10; do

CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path runs/dgcnn_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --iid ${iid} --severity ${sev} --corruption ${cor} --output ./modelnetc_our_${iid}/${model}_none_${cor}_${sev}.txt

done
done
done
done




# analytical or random initialized per-point embedding
for iid in 1 2 3 4 5; do
if [ ! -d "modelnetc_our_"${iid} ]; then
    mkdir "modelnetc_our_"${iid}
fi

for model in 'pctc' 'pctcmean' 'peat' 'peatmean' 'pointnetmean' 'pointnet' 'pointnetmlp3mean' ; do #
for cor in 'gaussian' 'uniform' 'ball_l' 'ball_m' 'ball_h' 'background' 'impulse' 'upsampling'; do #
for sev in 1 2 3 4 5 6 7 8 9 10; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_${model}_run_1_random/model_best_test.pth --exp-config configs/corruption/${model}.yaml --iid ${iid} --severity ${sev} --corruption ${cor} --output ./modelnetc_our_${iid}/${model}+random_none_${cor}_${sev}.txt

done
done
done
done