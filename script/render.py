# -*- coding: utf-8 -*-
"""
author:     zhikang.zeng
time  :     2025-01-09 11:12
"""
import argparse
import os
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Depth Render")
parser.add_argument('--img_dir', type=str, default=r'', help='render multiple frames of images')
parser.add_argument('--img_path', type=str, default=r'', help='render a single frame image')
parser.add_argument('--img_type', type=str, default=r'depth', help='optional: [depth, disp]')
parser.add_argument('--min_disp', type=float, default=2.0, help='min disp')
parser.add_argument('--max_disp', type=float, default=192.0, help='max disp')
parser.add_argument('--min_depth', type=float, default=0.0, help='min depth')
parser.add_argument('--max_depth', type=float, default=10000.0, help='max depth')
parser.add_argument('--save_dir', type=str, default=r'', help='directory to save results')
parser.add_argument('--save_gif', type=bool, default=False, help='save gif result')
parser.add_argument('--need_left_img', type=bool, default=False, help='gif image with left image')
parser.add_argument('--need_speckle_filter', type=bool, default=True, help='need speckle filter')
args = parser.parse_args()


# args.img_dir = r'D:\3_HoBot\3_RDK_X3_X5\14_Stereo\render\stereonet_images_zed2i_1'
# args.img_dir = r'D:\3_HoBot\3_RDK_X3_X5\14_Stereo\render\stereonet_images_zed2i_2'
# args.img_dir = r'D:\3_HoBot\3_RDK_X3_X5\14_Stereo\render\stereonet_images_zed2i_3'
# args.img_dir = r'D:\3_HoBot\3_RDK_X3_X5\14_Stereo\render\stereonet_images_zed2i_4'
# args.img_dir = r'D:\3_HoBot\3_RDK_X3_X5\14_Stereo\render\stereonet_images_s316_1'
# args.img_dir = r'D:\3_HoBot\3_RDK_X3_X5\14_Stereo\render\stereonet_images_s316_2'
# args.img_dir = r'D:\3_HoBot\3_RDK_X3_X5\14_Stereo\render\view_rock_data'
# args.img_type = 'disp'
# args.save_dir = os.path.split(args.img_dir)[0] + fr'\render_{args.img_type}_' + os.path.split(args.img_dir)[1]
# args.save_gif = True
# args.need_left_img = True


def is_cv16uc1(image):
    # 检查图像数据类型和通道数
    return image.dtype == np.uint16 and len(image.shape) == 2


def is_cv32fc1(image):
    # 检查图像数据类型和通道数
    return image.dtype == np.float32 and len(image.shape) == 2


if __name__ == '__main__':
    img_dir = args.img_dir
    img_path = args.img_path
    img_type = args.img_type
    min_disp = args.min_disp
    max_disp = args.max_disp
    min_depth = args.min_depth
    max_depth = args.max_depth
    save_dir = args.save_dir
    save_gif = args.save_gif
    need_left_img = args.need_left_img
    need_speckle_filter = args.need_speckle_filter
    print('=> args: ')
    print(f'       img_dir: {img_dir}')
    print(f'       img_path: {img_path}')
    print(f'       img_type: {img_type}')
    print(f'       min_disp: {min_disp}')
    print(f'       max_disp: {max_disp}')
    print(f'       min_depth: {min_depth}')
    print(f'       max_depth: {max_depth}')
    print(f'       save_dir: {save_dir}')
    print(f'       save_gif: {save_gif}')
    print(f'       need_left_img: {need_left_img}')
    print(f'       need_speckle_filter: {need_speckle_filter}')
    try:
        assert img_type in ['depth', 'disp']
    except:
        print('=> img_type needs to be set to [depth, disp]')

    waitkey_time = 200
    if img_path != '':
        img_path_list = [img_path]
        waitkey_time = 0
    elif img_dir != '':
        img_path_list = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if
                         filename.startswith(img_type) or filename.endswith('.tiff')]
    else:
        print('=> please enter image path!')
        exit(0)

    print(f'=> img count: {len(img_path_list)}')
    gif_frames = []
    for org_img_path in img_path_list:
        print('=============================================================')
        if not os.path.exists(org_img_path):
            print(f'=> {org_img_path} not exist')
            continue
        if not org_img_path.endswith(('.png', '.pfm', '.tiff')): continue
        # read img
        print(f'=> process {org_img_path}')
        org_img = cv2.imread(org_img_path, cv2.IMREAD_UNCHANGED)
        print(f'=> org_img [min, max]: [{org_img.min():.2f}, {org_img.max():.2f}]')

        # Limit the max and min values
        if img_type == 'disp':
            if not is_cv32fc1(org_img):
                print('=> disparity image format error!')
                continue
            org_img[org_img < min_disp] = 0
            org_img[org_img > max_depth] = 0
        if img_type == 'depth':
            if not is_cv16uc1(org_img):
                print('=> depth image format error!')
                continue
            org_img[org_img < min_depth] = 0
            org_img[org_img > max_depth] = 0
        print(f'=> limit org_img [min, max]: [{org_img.min():.2f}, {org_img.max():.2f}]')

        # speckle filter
        if need_speckle_filter:
            norm_img = cv2.normalize(org_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.filterSpeckles(norm_img, newVal=0, maxSpeckleSize=10, maxDiff=3)
            _, mask = cv2.threshold(norm_img, 0, 1, cv2.THRESH_BINARY)
            org_img *= mask

        # sort data
        flattened_data = org_img.flatten()
        sorted_data = np.sort(flattened_data[flattened_data > 0])
        # calc percentile
        percentile1 = np.percentile(sorted_data, 10)
        percentile2 = np.percentile(sorted_data, 50)
        percentile3 = np.percentile(sorted_data, 90)
        print(f'=> percentile: [{percentile1:.2f}, {percentile2:.2f}, {percentile3:.2f}]')
        # split data
        values1 = flattened_data[flattened_data <= percentile1]
        values2 = flattened_data[(flattened_data > percentile1) & (flattened_data <= percentile2)]
        values3 = flattened_data[(flattened_data > percentile2) & (flattened_data <= percentile3)]
        values4 = flattened_data[flattened_data > percentile3]
        # norm data
        # Jet伪彩色映射（0到1范围）的颜色段：
        # 0.0 - 0.25：从深蓝色逐渐过渡到浅蓝色
        # 0.25 - 0.5：从浅蓝色逐渐过渡到青绿色
        # 0.5 - 0.75：从青绿色逐渐过渡到黄色
        # 0.75 - 1.0：从黄色逐渐过渡到红色
        values1_norm = (values1 - np.min(values1)) / (percentile1 - np.min(values1)) * 0.25
        values2_norm = 0.25 + (values2 - percentile1) / (percentile2 - percentile1) * 0.25
        values3_norm = 0.5 + (values3 - percentile2) / (percentile3 - percentile2) * 0.25
        values4_norm = 0.75 + (values4 - percentile3) / (np.max(values4) - percentile3) * 0.25
        # merge norm data
        norm_data = np.zeros_like(flattened_data, dtype=float)
        norm_data[flattened_data <= percentile1] = values1_norm
        norm_data[(flattened_data > percentile1) & (flattened_data <= percentile2)] = values2_norm
        norm_data[(flattened_data > percentile2) & (flattened_data <= percentile3)] = values3_norm
        norm_data[flattened_data > percentile3] = values4_norm
        # revert to the shape of the org_img
        norm_data = norm_data.reshape(org_img.shape)

        # render
        colormap = plt.cm.jet
        colored_img = colormap(norm_data)
        colored_img[org_img == 0] = (0, 0, 0, 1)
        colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
        colored_img = cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR)
        if need_left_img:
            filepath, filename = os.path.split(org_img_path)
            filename_without_extension, _ = os.path.splitext(filename)
            left_filename = filename_without_extension.replace(img_type, f'left') + '.png'
            left_filepath = os.path.join(filepath, left_filename)
            if os.path.exists(left_filepath):
                left_img = cv2.imread(left_filepath, cv2.IMREAD_COLOR)
                colored_img = np.vstack((left_img, colored_img))
        cv2.imshow("render img", colored_img)
        cv2.waitKey(waitkey_time)

        if save_dir != '':
            os.makedirs(save_dir, exist_ok=True)
            filepath, filename = os.path.split(org_img_path)
            filename_without_extension, _ = os.path.splitext(filename)
            result_filename = filename_without_extension.replace(img_type, f'render_{img_type}') + '.png'
            result_filepath = os.path.join(save_dir, result_filename)
            colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
            imageio.imwrite(result_filepath, colored_img)
            print(f'=> save render result to {result_filepath}')

            if save_gif:
                gif_frames.append(colored_img)

    if os.path.exists(save_dir) and save_gif and len(gif_frames) > 2:
        print('=============================================================')
        result_filepath = os.path.join(save_dir, 'result.gif')
        print(f'=> save gif result to {result_filepath}')
        imageio.mimsave(result_filepath, gif_frames, fps=5, loop=0)
