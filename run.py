from main_test_swinir import main
import argparse

args = argparse.Namespace(
    task='real_sr',
    scale=4,
    noise=15,
    jpeg=40,
    training_patch_size=128,
    large_model=True,
    model_path='model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
    folder_lq='testsets/RealSRSet+5images',
    folder_gt=None,
    tile=None,
    tile_overlap=32
)

main(args)
