python3 tools/train.py \
    -c ./ppcls/configs/quick_start/ResNet50_vd.yaml \
    -o Arch.pretrained=False \
    -o Global.device=cpu

py -3 tools/train.py -c ./ppcls/configs/quick_start/ResNet50_vd.yaml -o Arch.pretrained=False -o Global.device=cpu