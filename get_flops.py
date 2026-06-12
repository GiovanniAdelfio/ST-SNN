from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from pyskl.models import build_model
import torch
import argparse

def main():
    # 1. Inerst the 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    args = parser.parse_args()
    config_path = args.config

    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model)
    model.eval()


    def custom_forward(keypoint):
        feat = model.extract_feat(keypoint)
        return model.cls_head(feat)
        
    model.forward = custom_forward
    # --------------------------

    # Ordine input: (Num_Subjects, Channels, Frames, Joints)

    input_shape = (2, 100, 25, 3)

    print(f"Calcolando FLOPs per input shape: {input_shape}...")
    
    flops, params = get_model_complexity_info(
        model, 
        input_shape, 
        print_per_layer_stat=False
    )

    print("=" * 30)
    print(f"Total FLOPs: {flops}")
    print(f"Total Params: {params}")
    print("=" * 30)

if __name__ == '__main__':
    main()