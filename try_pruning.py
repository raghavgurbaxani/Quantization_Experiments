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

def predict(model):
    model.eval()
    img_path_list = preprossing.get_images(cfg.test_img_path)
    for index in range(len(img_path_list)):
        im_fn = img_path_list[index]
        im = cv2.imread(im_fn)[:,:,::-1]
        if im is None:
            print("can not find image of %s" % (im_fn))
            continue

        print('EAST <==> TEST <==> epoch:{}, idx: <==> Begin'.format( index))
        im_resized, (ratio_h, ratio_w) = utils.resize_image(im)
        im_resized = im_resized.astype(np.float32)
        im_resized = im_resized.transpose(2, 0, 1)
        im_tensor = torch.from_numpy(im_resized)
        im_tensor = im_tensor #.cuda()
        im_tensor = im_tensor.unsqueeze(0)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()

        score, geometry = model(im_tensor)

        timer['net'] = time.time() - start
        score = score.permute(0, 2, 3, 1)
        geometry = geometry.permute(0, 2, 3, 1)
        score = score.data.cpu().numpy()
        geometry = geometry.data.cpu().numpy()
        boxes, timer = utils.detect(score_map=score, geo_map=geometry, timer=timer,
                                    score_map_thresh=cfg.score_map_thresh, box_thresh=cfg.box_thresh,
                                    nms_thres=cfg.box_thresh)
        print('EAST <==> TEST <==> idx:{} <==> model:{:.2f}ms, restore:{:.2f}ms, nms:{:.2f}ms'
              .format(index, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

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

        if cfg.write_images:
            img_path = os.path.join(cfg.res_img_path, os.path.basename(im_fn))
            cv2.imwrite(img_path, im[:,:,::-1])
            print('EAST <==> TEST <==> Save image at:{} <==> Done'.format(img_path))

        print('EAST <==> TEST <==> Record and Save <==> epoch:, ids:{} <==> Done'.format(index))

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
    
    print('Original Size:')
    print_size_of_model(model)
    
    ###############
    
    print('Try pruning')
    import torch.nn.utils.prune as prune
    for name, module in model.named_modules():
    # prune 40% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.4)
            #prune.l1_unstructured(module, name='bias', amount=0.3)
            prune.remove(module, 'weight')
            #prune.remove(module, 'bias')
    # prune 40% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)
            #prune.l1_unstructured(module, name='bias', amount=0.4)
            prune.remove(module, 'weight')
            #prune.remove(module, 'bias')
    #print(dict(model.named_buffers()).keys())
    print_size_of_model(model)
   

if __name__ == "__main__":
    main()

img_path_list = preprossing.get_images(cfg.test_img_path)
