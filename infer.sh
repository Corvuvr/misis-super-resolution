#!/bin/bash

RT4KSR=false
ESRGAN=false
METRIC=false

for i in "$@"; do
  case $i in
        --RT4KSR)
        RT4KSR=true 
        ;;
        --ESRGAN)
        ESRGAN=true
        ;;
        --metrics)
        METRIC=true
        ;;
        -*|--*)
        echo "Unknown option $i"
        exit 1
        ;;
        *)
        ;;
    esac
done

# Downscale DIV2K
conda activate CorvuvrESRGAN
    python downscale.py -i datasets/DIV2K/DIV2K_valid_HR -o datasets/DIV2K/DIV2K -s 2
conda deactivate

if $RT4KSR ; then
echo "Running RT4KSR"
conda activate CorvuvrRT4KSR
    cd RT4KSR
        python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/DIV2K/DIV2K/
        python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/General100/General100   
        python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/BSDS100/BSDS100         
        python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/urban100/urban100       
        python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/set14/set14/set14       
    cd -
conda deactivate
fi


if $ESRGAN ; then
echo "Running ESRGAN"
conda activate CorvuvrESRGAN
    export LD_LIBRARY_PATH=$(dirname $(ldconfig -p | grep libcuda.so | cut -d '>' -f2)):$LD_LIBRARY_PATH
    cd Real-ESRGAN
        python inference_realesrgan.py -n RealESRGAN_x4plus --outscale 2 --pre_downscale 2 --suffix "out" -i ../datasets/DIV2K/DIV2K -o results/DIV2K
        python inference_realesrgan.py -n RealESRGAN_x4plus --outscale 2 --pre_downscale 2 --suffix "out" -i ../datasets/General100/General100 -o results/General100
        python inference_realesrgan.py -n RealESRGAN_x4plus --outscale 2 --pre_downscale 2 --suffix "out" -i ../datasets/BSDS100/BSDS100 -o results/BSDS100 
        python inference_realesrgan.py -n RealESRGAN_x4plus --outscale 2 --pre_downscale 2 --suffix "out" -i ../datasets/urban100/urban100 -o results/urban100
        python inference_realesrgan.py -n RealESRGAN_x4plus --outscale 2 --pre_downscale 2 --suffix "out" -i ../datasets/set14/set14/set14 -o results/set14 
    cd - 
conda deactivate
fi

if $METRIC ; then
conda activate CorvuvrStudy
    python study/plotmaker.py
conda deactivate
fi