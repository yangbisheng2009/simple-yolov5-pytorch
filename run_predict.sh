CUDA_VISIBLE_DEVICES=0 python predict.py --checkpoint checkpoints/silkworm-m/111_0.9712_0.7970_0.8145.pth -p configs/silkworm/silkworm-m.yaml --input-images /home/yangfg/work/corpus/silkworm/voc/JPEGImages/ --output-images outputs/ --agnostic-nms
