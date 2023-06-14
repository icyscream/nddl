
import os, sys
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
np.random.seed(0)

def main(cfg):
    # creat folders 
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    #指令输入的路径，例如：python main_train.py --cfg configs/release_version/deca_pretrain.yml 
    shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))
    
    # cudnn related setting
    #根据硬件环境自动选择最优的卷积算法，以获得最佳性能
    cudnn.benchmark = True
    #关闭cuDNN的确定性模式，使用非确定性的随机算法继续宁计算，以增加模型的随机性，使其更具有泛化能力
    torch.backends.cudnn.deterministic = False
    #启用cuDNN库，一遍在gpu上加速深度学习模型的计算。
    torch.backends.cudnn.enabled = True

    # start training
    # deca model
    from decalib.deca import DECA
    from decalib.trainer import Trainer
    #是一个将三维场景转换为二维图像的程序或库
    cfg.rasterizer_type = 'pytorch3d'
    #创建实例
    deca = DECA(cfg)
    #
    trainer = Trainer(model=deca, config=cfg)

    ## start train
    trainer.fit()

if __name__ == '__main__':
    from decalib.utils.config import parse_args
    cfg = parse_args()
    if cfg.cfg_file is not None:
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.exp_name = exp_name
    main(cfg)

# run:
# python main_train.py --cfg configs/release_version/deca_pretrain.yml 