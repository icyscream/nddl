
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
#进度条库
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images
    #初始化读取路径内的图片或者视频，并放入init方法内的imagepath_list中
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step, device=device)

    # run DECA
    #使用flame纹理模型来处理uv纹理贴图
    deca_cfg.model.use_tex = args.useTex
    #原图像大小呈现结果
    deca_cfg.rasterizer_type = args.rasterizer_type
    #从输入图像中提取纹理作为uv纹理映射
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]
        #当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict) #tensor
            if args.render_orig:
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = testdata[i]['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
                orig_visdict['inputs'] = original_image            

        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
            if args.render_orig:
                cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
                if args.render_orig:
                    image = util.tensor2image(orig_visdict[vis_name][0])
                    cv2.imwrite(os.path.join(savefolder, name, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='测试数据的路径，可以是图像文件夹、图像路径、图像列表、视频')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='输出目录的路径，结果(obj, TXT文件)将存储在其中。')
    parser.add_argument('--device', default='cuda', type=str,
                        help='设置设备，CPU使用CPU' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='是否裁剪输入图像，仅当测试图像裁剪良好时设置为false' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='每个步骤的视频数据示例图像' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='用于裁剪脸部的检测器，查看decalibdetectors.py了解详细信息' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='光栅类型:pytorch3d或标准' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='是否以原始图像大小呈现结果，目前仅当rasterizer_type=standard时有效')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='是否使用FLAME纹理模型来生成uv纹理贴图，只在下载纹理模型时设置为True' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='是否从输入图像中提取纹理作为uv纹理映射，如果你想从FLAME模式中提取反照贴图，则设置为false' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='是否保存输出的可视化' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='是否保存2D和3D关键点' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='是否保存深度图像' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='是否将输出保存为.obj, detail mesh将以_detail.obj结束。\注意，保存objs可能会很慢' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='是否将输出保存为.mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='是否将可视化输出保存为单独的图像' )
    main(parser.parse_args())