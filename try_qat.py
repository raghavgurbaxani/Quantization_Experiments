import os
import config as cfg
from model import East
import torch
import utils
import preprossing
import cv2
import numpy as np
import time
import loss

def uninplace(model):
    if hasattr(model, 'inplace'):
        model.inplace = False
    if not model.children():
        return
    for child in model.children():
        uninplace(child)
        
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def predict(model, epoch):

    model.eval()
    img_path_list = preprossing.get_images(cfg.test_img_path)

    for index in range(len(img_path_list)):

        im_fn = img_path_list[index]
        im = cv2.imread(im_fn)[:,:,::-1]
        if im is None:
            print("can not find image of %s" % (im_fn))
            continue

        print('EAST <==> TEST <==> epoch:{}, idx:{} <==> Begin'.format(epoch, index))
        # 图像进行放缩
        im_resized, (ratio_h, ratio_w) = utils.resize_image(im)
        im_resized = im_resized.astype(np.float32)
        # 图像转换成tensor格式
        im_resized = im_resized.transpose(2, 0, 1)
        im_tensor = torch.from_numpy(im_resized)
        im_tensor = im_tensor #.cuda()
        # 图像数据增加一维
        im_tensor = im_tensor.unsqueeze(0)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()

        # 输入网络进行推断
        score, geometry = model(im_tensor)

        timer['net'] = time.time() - start
        # score与geometry转换成numpy格式
        score = score.permute(0, 2, 3, 1)
        geometry = geometry.permute(0, 2, 3, 1)
        score = score.data.cpu().numpy()
        geometry = geometry.data.cpu().numpy()
        # 文本框检测
        boxes, timer = utils.detect(score_map=score, geo_map=geometry, timer=timer,
                                    score_map_thresh=cfg.score_map_thresh, box_thresh=cfg.box_thresh,
                                    nms_thres=cfg.box_thresh)
        print('EAST <==> TEST <==> idx:{} <==> model:{:.2f}ms, restore:{:.2f}ms, nms:{:.2f}ms'
              .format(index, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        # save to txt file
        if boxes is not None:
            res_file = os.path.join(
                cfg.res_img_path,
                'res_{}.txt'.format(
                    os.path.basename(im_fn).split('.')[0]))

            with open(res_file, 'w') as f:
                for box in boxes:
                    # to avoid submitting errors
                    box = utils.sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    ))
                    cv2.polylines(im[:,:,::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(255, 255, 0), thickness=1)
                print('EAST <==> TEST <==> Save txt   at:{} <==> Done'.format(res_file))

        # 图片输出
        if cfg.write_images:
            img_path = os.path.join(cfg.res_img_path, os.path.basename(im_fn))
            cv2.imwrite(img_path, im[:,:,::-1])
            print('EAST <==> TEST <==> Save image at:{} <==> Done'.format(img_path))

        print('EAST <==> TEST <==> Record and Save <==> epoch:{}, ids:{} <==> Done'.format(epoch, index))

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    weight_loss = utils.Regularization(model, cfg.l2_weight_decay, p=2)
    cnt = 0
    for i, (img, img_path, score_map, geo_map, training_mask) in enumerate(data_loader):       
        cnt += 1
        img, score_map, geo_map, training_mask = img.to(device),score_map.to(device),geo_map.to(device),training_mask.to(device)
        f_score, f_geometry = model(img)
        model_loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
        total_loss = model_loss + weight_loss(model)       
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()              
        if i % cfg.print_freq == 0:
            print(total_loss.item())
    return


def main():
    # prepare output directory
    # global epoch
    print('EAST <==> TEST <==> Create Res_file and Img_with_box <==> Begin')
    result_root = os.path.abspath(cfg.res_img_path)
    if not os.path.exists(result_root):
        os.mkdir(result_root)

    print('EAST <==> Prepare <==> Network <==> Begin')
    model = East()
    model = torch.nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model #.cuda()
    # 载入模型
    if os.path.isfile(cfg.checkpoint):
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(cfg.checkpoint))
        checkpoint = torch.load(cfg.checkpoint,map_location='cpu')
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Done".format(cfg.checkpoint))
    else:
        print('Can not find checkpoint !!!')
        exit(1)
    print()
    print('###############')
    print()
    example = torch.rand(1, 3, 224, 224)
    
    #traced_script_module = torch.jit.trace(model, example)
    uninplace(model)
    l1=[['module.conv1','module.bn1','module.relu1'],['module.conv2','module.bn2','module.relu2'],['module.conv3','module.bn3','module.relu3'],['module.conv4','module.bn4','module.relu4'],['module.conv5','module.bn5','module.relu5'],['module.conv6','module.bn6','module.relu6'],['module.conv7','module.bn7','module.relu7']]
    #s4
    l2=[['module.s4.0.conv.0','module.s4.0.conv.1','module.s4.0.conv.2'],['module.s4.0.conv.3','module.s4.0.conv.4','module.s4.0.conv.5'],['module.s4.0.conv.6','module.s4.0.conv.7'],['module.s4.1.conv.0','module.s4.1.conv.1','module.s4.1.conv.2'],['module.s4.1.conv.3','module.s4.1.conv.4','module.s4.1.conv.5'],['module.s4.1.conv.6','module.s4.1.conv.7'],['module.s4.2.conv.0','module.s4.2.conv.1','module.s4.2.conv.2'],['module.s4.2.conv.3','module.s4.2.conv.4','module.s4.2.conv.5'],['module.s4.2.conv.6','module.s4.2.conv.7']] 
    #s3
    l3=[['module.s3.0.conv.0','module.s3.0.conv.1','module.s3.0.conv.2'],['module.s3.0.conv.3','module.s3.0.conv.4','module.s3.0.conv.5'],['module.s3.0.conv.6','module.s3.0.conv.7'],['module.s3.1.conv.0','module.s3.1.conv.1','module.s3.1.conv.2'],['module.s3.1.conv.3','module.s3.1.conv.4','module.s3.1.conv.5'],['module.s3.1.conv.6','module.s3.1.conv.7'],['module.s3.2.conv.0','module.s3.2.conv.1','module.s3.2.conv.2'],['module.s3.2.conv.3','module.s3.2.conv.4','module.s3.2.conv.5'],['module.s3.2.conv.6','module.s3.2.conv.7'],['module.s3.3.conv.0','module.s3.3.conv.1','module.s3.3.conv.2'],['module.s3.3.conv.3','module.s3.3.conv.4','module.s3.3.conv.5'],['module.s3.3.conv.6','module.s3.3.conv.7'],['module.s3.4.conv.0','module.s3.4.conv.1','module.s3.4.conv.2'],['module.s3.4.conv.3','module.s3.4.conv.4','module.s3.4.conv.5'],['module.s3.4.conv.6','module.s3.4.conv.7'],['module.s3.5.conv.0','module.s3.5.conv.1','module.s3.5.conv.2'],['module.s3.5.conv.3','module.s3.5.conv.4','module.s3.5.conv.5'],['module.s3.5.conv.6','module.s3.5.conv.7'],['module.s3.6.conv.0','module.s3.6.conv.1','module.s3.6.conv.2'],['module.s3.6.conv.3','module.s3.6.conv.4','module.s3.6.conv.5'],['module.s3.6.conv.6','module.s3.6.conv.7']] 
    #s2
    l4=[['module.s2.0.conv.0','module.s2.0.conv.1','module.s2.0.conv.2'],['module.s2.0.conv.3','module.s2.0.conv.4','module.s2.0.conv.5'],['module.s2.0.conv.6','module.s2.0.conv.7'],['module.s2.1.conv.0','module.s2.1.conv.1','module.s2.1.conv.2'],['module.s2.1.conv.3','module.s2.1.conv.4','module.s2.1.conv.5'],['module.s2.1.conv.6','module.s2.1.conv.7'],['module.s2.2.conv.0','module.s2.2.conv.1','module.s2.2.conv.2'],['module.s2.2.conv.3','module.s2.2.conv.4','module.s2.2.conv.5'],['module.s2.2.conv.6','module.s2.2.conv.7']] 
    #s1
    l5=[['module.s1.0.0','module.s1.0.1','module.s1.0.2'],['module.s1.1.conv.0','module.s1.1.conv.1','module.s1.1.conv.2'],['module.s1.1.conv.3','module.s1.1.conv.4'],['module.s1.2.conv.0','module.s1.2.conv.1','module.s1.2.conv.2'],['module.s1.2.conv.3','module.s1.2.conv.4','module.s1.2.conv.5'],['module.s1.2.conv.6','module.s1.2.conv.7']]
    #( s1 - 1 and 2)
    l6=[['module.mobilenet.features.0.0','module.mobilenet.features.0.1','module.mobilenet.features.0.2'],['module.mobilenet.features.1.conv.0','module.mobilenet.features.1.conv.1','module.mobilenet.features.1.conv.2'],['module.mobilenet.features.1.conv.3','module.mobilenet.features.1.conv.4']]

    l7=[['module.mobilenet.features.2.conv.0','module.mobilenet.features.2.conv.1','module.mobilenet.features.2.conv.2'],['module.mobilenet.features.2.conv.3','module.mobilenet.features.2.conv.4','module.mobilenet.features.2.conv.5'],['module.mobilenet.features.2.conv.6','module.mobilenet.features.2.conv.7'],['module.mobilenet.features.3.conv.0','module.mobilenet.features.3.conv.1','module.mobilenet.features.3.conv.2'],['module.mobilenet.features.3.conv.3','module.mobilenet.features.3.conv.4','module.mobilenet.features.3.conv.5'],['module.mobilenet.features.3.conv.6','module.mobilenet.features.3.conv.7'],['module.mobilenet.features.4.conv.0','module.mobilenet.features.4.conv.1','module.mobilenet.features.4.conv.2'],['module.mobilenet.features.4.conv.3','module.mobilenet.features.4.conv.4','module.mobilenet.features.4.conv.5'],['module.mobilenet.features.4.conv.6','module.mobilenet.features.4.conv.7'],['module.mobilenet.features.5.conv.0','module.mobilenet.features.5.conv.1','module.mobilenet.features.5.conv.2'],['module.mobilenet.features.5.conv.3','module.mobilenet.features.5.conv.4','module.mobilenet.features.5.conv.5'],['module.mobilenet.features.5.conv.6','module.mobilenet.features.5.conv.7'],['module.mobilenet.features.6.conv.0','module.mobilenet.features.6.conv.1','module.mobilenet.features.6.conv.2'],['module.mobilenet.features.6.conv.3','module.mobilenet.features.6.conv.4','module.mobilenet.features.6.conv.5'],['module.mobilenet.features.6.conv.6','module.mobilenet.features.6.conv.7'],['module.mobilenet.features.7.conv.0','module.mobilenet.features.7.conv.1','module.mobilenet.features.7.conv.2'],['module.mobilenet.features.7.conv.3','module.mobilenet.features.7.conv.4','module.mobilenet.features.7.conv.5'],['module.mobilenet.features.7.conv.6','module.mobilenet.features.7.conv.7'],['module.mobilenet.features.8.conv.0','module.mobilenet.features.8.conv.1','module.mobilenet.features.8.conv.2'],['module.mobilenet.features.8.conv.3','module.mobilenet.features.8.conv.4','module.mobilenet.features.8.conv.5'],['module.mobilenet.features.8.conv.6','module.mobilenet.features.8.conv.7'],['module.mobilenet.features.9.conv.0','module.mobilenet.features.9.conv.1','module.mobilenet.features.9.conv.2'],['module.mobilenet.features.9.conv.3','module.mobilenet.features.9.conv.4','module.mobilenet.features.9.conv.5'],['module.mobilenet.features.9.conv.6','module.mobilenet.features.9.conv.7'],['module.mobilenet.features.10.conv.0','module.mobilenet.features.10.conv.1','module.mobilenet.features.10.conv.2'],['module.mobilenet.features.10.conv.3','module.mobilenet.features.10.conv.4','module.mobilenet.features.10.conv.5'],['module.mobilenet.features.10.conv.6','module.mobilenet.features.10.conv.7'],['module.mobilenet.features.11.conv.0','module.mobilenet.features.11.conv.1','module.mobilenet.features.11.conv.2'],['module.mobilenet.features.11.conv.3','module.mobilenet.features.11.conv.4','module.mobilenet.features.11.conv.5'],['module.mobilenet.features.11.conv.6','module.mobilenet.features.11.conv.7'],['module.mobilenet.features.12.conv.0','module.mobilenet.features.12.conv.1','module.mobilenet.features.12.conv.2'],['module.mobilenet.features.12.conv.3','module.mobilenet.features.12.conv.4','module.mobilenet.features.12.conv.5'],['module.mobilenet.features.12.conv.6','module.mobilenet.features.12.conv.7'],['module.mobilenet.features.13.conv.0','module.mobilenet.features.13.conv.1','module.mobilenet.features.13.conv.2'],['module.mobilenet.features.13.conv.3','module.mobilenet.features.13.conv.4','module.mobilenet.features.13.conv.5'],['module.mobilenet.features.13.conv.6','module.mobilenet.features.13.conv.7'],['module.mobilenet.features.14.conv.0','module.mobilenet.features.14.conv.1','module.mobilenet.features.14.conv.2'],['module.mobilenet.features.14.conv.3','module.mobilenet.features.14.conv.4','module.mobilenet.features.14.conv.5'],['module.mobilenet.features.14.conv.6','module.mobilenet.features.14.conv.7'],['module.mobilenet.features.15.conv.0','module.mobilenet.features.15.conv.1','module.mobilenet.features.15.conv.2'],['module.mobilenet.features.15.conv.3','module.mobilenet.features.15.conv.4','module.mobilenet.features.15.conv.5'],['module.mobilenet.features.15.conv.6','module.mobilenet.features.15.conv.7'],['module.mobilenet.features.16.conv.0','module.mobilenet.features.16.conv.1','module.mobilenet.features.16.conv.2'],['module.mobilenet.features.16.conv.3','module.mobilenet.features.16.conv.4','module.mobilenet.features.16.conv.5'],['module.mobilenet.features.16.conv.6','module.mobilenet.features.16.conv.7']]

    #modules_to_fuse=l1+l2+l3+l4+l5
    #print(model)
    print('Original Size:')
    print_size_of_model(model)
    print()
    fused_model = torch.quantization.fuse_modules(model, l7)
    fused_model = torch.quantization.fuse_modules(model, l6)
    fused_model = torch.quantization.fuse_modules(model, l1+l2+l3+l4+l5)
    
    print('Fused model Size:')
    print_size_of_model(fused_model)
    print()
    #print(fused_model)
    
    #fused_model.qconfig = torch.quantization.QConfig(activation=torch.quantization.default_histogram_observer,weight=torch.quantization.default_per_channel_weight_observer)
    fused_model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(fused_model, inplace=True)
    
    
    from data_loader import custom_dset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    trainset = custom_dset(transform=transforms.ToTensor())
    device=torch.device('cpu')
    train_loader = DataLoader(trainset, batch_size=cfg.train_batch_size_per_gpu * cfg.gpu,shuffle=True, num_workers=0)
    for i, (img, img_path, score_map, geo_map, training_mask) in enumerate(train_loader):
        img, score_map, geo_map, training_mask = img.to(device),score_map.to(device),geo_map.to(device),training_mask.to(device)
        f_score, f_geometry = fused_model(img)
        
    quantized = torch.quantization.convert(fused_model, inplace=False)
    print('Quantized model Size:')
    print_size_of_model(quantized)
    
    num_train_batches = 20
    print('***QAT***')
    print()
    criterion = loss.LossFunc()
    pre_params = list(map(id, model.module.mobilenet.parameters()))
    post_params = filter(lambda p: id(p) not in pre_params, model.module.parameters())
    
    optimizer = torch.optim.Adam([{'params': model.module.mobilenet.parameters(), 'lr': cfg.pre_lr},
                                  {'params': post_params, 'lr': cfg.lr}])
    fused_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # Train and check accuracy after each epoch
    for nepoch in range(8):
        train_one_epoch(fused_model, criterion, optimizer, train_loader, torch.device('cpu'), num_train_batches)
        if nepoch > 3:
            # Freeze quantizer parameters
            fused_model.apply(torch.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            fused_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    quantized_model = torch.quantization.convert(fused_model.eval(), inplace=False)
    
    print('QAT model Size:')
    print_size_of_model(quantized_model)
    print('Done')
    print(quantized)
    
    #print(quantized)
    #q_example = torch.quantize_per_tensor(example, scale=1e-3, zero_point=128,dtype=torch.quint8)
    #predict(quantized, epoch)   
    #import sys
    #sys.stdout = open("quantized_model.txt", "w")
    #print(quantized)
    #sys.stdout.close()
    

if __name__ == "__main__":
    main()

img_path_list = preprossing.get_images(cfg.test_img_path)
