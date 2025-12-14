CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        torchrun --nproc_per_node=8 --master_port=10210 train.py --local_rank 0 \
            --config-file ./checkpoints/kinetics-gebd/config.yaml \
            --gebd_data_dir /your_kinetics_folder/ \
            --test-only \
            --resume ./checkpoints/kinetics-gebd/model_best.pth \
            --seed 42 \
            SOLVER.BATCH_SIZE 2 \
            DIFFUSION.SAMPLING_TIMESTEPS 32 \
            DIFFUSION.CFG_SCALE 7.0 \
