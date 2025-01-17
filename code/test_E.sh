
#    --resume checkpoints/FlowNet2-S_checkpoint.pth.tar
#    --resume work/FlowNet2S_model_best.pth.tar

python main.py \
    --inference \
    --model FlowNetE \
    --loss=EPELoss \
    --save_flow \
    --inference_visualize \
    --inference_dataset MpiSintelClean \
    --inference_dataset_root data/validation \
    --resume work/FlowNetE_model_best.pth.tar

# python main.py --inference --model FlowNetE --save_flow --inference_visualize --inference_dataset MpiSintelClean --inference_dataset_root data/validation --resume work/FlowNetE_model_best.pth.tar
