# evaluation on the original ModelNet40-C dataset


# models trained end-to-end
if [ ! -d "modelnetc"]; then
    mkdir "modelnetc"
fi

for model in  'pct' 'pctc' 'pctcmean' 'peat' 'peatmean' 'rscnn' 'pointnet2' 'dgcnn' 'pemax' 'pemean' 'pemedian' 'pointnetmean' 'pointnet' 'pointnetmlp3mean'; do 
for cor in 'uniform' 'gaussian' 'background' 'impulse' 'upsampling' 'distortion_rbf' 'distortion_rbf_inv' 'density' 'density_inc' 'shear' 'rotation' 'cutout' 'distortion'  'occlusion' 'lidar'; do
for sev in 1 2 3 4 5; do

CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path runs/dgcnn_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --iid 0 --severity ${sev} --corruption ${cor} --output ./modelnetc/${model}_none_${cor}_${sev}.txt

done
done
done




# analytical or random initialized per-point embedding
if [ ! -d "modelnetc" ]; then
    mkdir "modelnetc"
fi

for model in 'pctc' 'pctcmean' 'peat' 'peatmean' 'pointnetmean' 'pointnet' 'pointnetmlp3mean'; do #
for cor in 'uniform' 'gaussian' 'background' 'impulse' 'upsampling' 'distortion_rbf' 'distortion_rbf_inv' 'density' 'density_inc' 'shear' 'rotation' 'cutout' 'distortion'  'occlusion' 'lidar'; do
for sev in 1 2 3 4 5; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_${model}_run_1_random/model_best_test.pth --exp-config configs/corruption/${model}.yaml --iid 0 --severity ${sev} --corruption ${cor} --output ./modelnetc/${model}+random_none_${cor}_${sev}.txt

done
done
done