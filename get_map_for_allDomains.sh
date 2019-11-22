#!/bin/bash
domain=("dog" "gymnastics" "parkour" "skating" "skiing" "surfing")
for i in "${domain[@]}";do python inference.py --test  --domain $i  --pretrain_path './models/save_ContextFea_final_'$i'.pth'; done
