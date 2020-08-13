import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class PatchManifoldLoss(nn.Module):
    def __init__(self, alpha, split,filter):
        super(PatchManifoldLoss, self).__init__()
        self.alpha = alpha
        self.split = split
        self.filter = filter

    def forward(self, inputs, targets):
        self.pathmanifoldloss = self.coss_patchmaniflod(inputs,targets)
        return self.pathmanifoldloss * self.alpha

    def coss_patchmaniflod(self, inputs, targets):
        # resample and return h and w in patch
        targets, h, w = self.resample(targets, self.split)
        inputs, h, w = self.resample(inputs, self.split)

        h = int(h)
        w = int(w)
        #
        sample_h = inputs.shape[-2]
        sample_w = inputs.shape[-1]

        # expand the targets sample
        target_ex = self.expand(targets)
        input_ex = self.expand(inputs)

        map_input = torch.zeros([self.split ** 2, self.split ** 2]).cuda()
        map_target = torch.zeros([self.split ** 2, self.split ** 2]).cuda()

        #
        for i in range(self.split):
            for j in range(self.split):
                target = target_ex[:, : ,i*h:sample_h+i*h,j*w:sample_w+j*w]
                input = input_ex[:, : ,i*h:sample_h+i*h,j*w:sample_w+j*w]

                temp_target = torch.exp(-torch.pow((target - targets), 2))
                temp_input = torch.pow((input - inputs),2)
                # temp_input = torch.exp(-torch.pow((input - inputs),2))

                for n in range(self.split):
                    for m in range(self.split):
                        patch_input = temp_input[:,:, n*h:n*h+h,m*w:m*w+w]
                        pacth_target = temp_target[:,:, n*h:n*h+h,m*w:m*w+w]

                        map_input[n*self.split+m,((i+n)%self.split)*self.split+(j+m)%self.split] = torch.mean(patch_input)
                        map_target[n*self.split+m,((i+n)%self.split)*self.split+(j+m)%self.split] = torch.mean(pacth_target)

        for i in range(self.split * self.split):
            feature = map_target[i]
            feature = torch.sort(feature)
            for j in range(self.split - self.filter):
                ind = feature.indices[j]
                map_target[i,ind] = 0

        value = map_input * map_target

        return value.mean()

    def expand(self, targets):
        return targets.repeat(1,1,2,2)

    def resample(self,targets, split):
        h = targets.shape[-2] / split
        w = targets.shape[-1] / split

        h = int(h / 10)
        w = int(w / 10)

        reshape_h = h * split
        reshape_w = w * split

        x = F.interpolate(targets, size=[int(reshape_h),int(reshape_w)], mode='bilinear', align_corners=False)

        return x , h ,w


class PatchManifoldLossPre(nn.Module):
    def __init__(self, alpha, split ,filter):
        super(PatchManifoldLossPre, self).__init__()
        self.alpha = alpha
        self.split = split
        self.filer = filter

    def forward(self, prediction, weights):
        self.pathmanifoldloss = self.coss_patchmanifold2(prediction, weights)
        return self.pathmanifoldloss * self.alpha

    def coss_patchmanifold(self, prediction, weights):
        prediction, h, w = self.resample(prediction, self.split)
        h = int(h)
        w = int(w)

        map_prediction = torch.zeros(weights.shape).cuda()
        for b in range(weights.shape[0]):
            for i in range(weights.shape[1]):
                patch_prediction = prediction[b,:,int(i/self.split):int(i/self.split)+h, int(i%self.split):int(i%self.split)+w]
                for j in range(weights.shape[2]):
                    if weights[b,i,j] != 0:
                        obj_prediction = prediction[b,:,int(j/self.split):int(j/self.split)+h, int(j%self.split):int(j%self.split)+w]
                        map_prediction[b,i,j] = torch.mean(torch.pow((patch_prediction - obj_prediction), 2))

        value =  map_prediction*weights
        return value.mean()

    def coss_patchmanifold2(self, prediction, weights):
        prediction, h ,w = self.resample(prediction, self.split)
        h = int(h)
        w = int(w)

        sample_h = prediction.shape[-2]
        sample_w = prediction.shape[-1]

        # expand the targets sample
        # target_ex = self.expand(targets)
        input_ex = self.expand(prediction)

        map_input = torch.zeros([self.split ** 2, self.split ** 2]).cuda()
        map_target = torch.zeros([self.split ** 2, self.split ** 2]).cuda()

        #
        for i in range(self.split):
            for j in range(self.split):
                # target = target_ex[:, :, i * h:sample_h + i * h, j * w:sample_w + j * w]
                input = input_ex[:, :, i * h:sample_h + i * h, j * w:sample_w + j * w]

                # temp_target = torch.exp(-torch.pow((target - targets), 2))
                temp_input = torch.pow((input - prediction), 2)
                # temp_input = torch.exp(-torch.pow((input - inputs),2))

                for n in range(self.split):
                    for m in range(self.split):
                        patch_input = temp_input[:, :, n * h:n * h + h, m * w:m * w + w]
                        # pacth_target = temp_target[:, :, n * h:n * h + h, m * w:m * w + w]

                        map_input[n * self.split + m, ((i + n) % self.split) * self.split + (
                                    j + m) % self.split] = torch.mean(patch_input)
                        # map_target[n * self.split + m, ((i + n) % self.split) * self.split + (
                        #             j + m) % self.split] = torch.mean(pacth_target)
        value = map_input * weights
        return value.mean()

    def expand(self, targets):
        return targets.repeat(1,1,2,2)

    def resample(self,targets, split):
        h = targets.shape[-2] / split
        w = targets.shape[-1] / split

        h = int(h / 10)
        w = int(w / 10)

        reshape_h = h * split
        reshape_w = w * split

        x = F.interpolate(targets, size=[int(reshape_h),int(reshape_w)], mode='bilinear', align_corners=False)

        return x , h ,w




# if __name__ == '__main__':
#     input = torch.ones([8,19,5,5])
#     for i  in range(5):
#         for j in range(5):
#             input[:,:,i,j] = i * 5 + j
#     loss = coss_manifode(input)
#     # print(input)
#     # output = input[:,:,0:-1-2,:0:-1-2]
#     # temp = input[:,:,0:5-2,0:5-2] - output[:,:,0:5-2,0+1:5-1]
#     print(loss)
#     # print(loss)
#     # print(temp)
#     # print(temp.size())
#     # print(loss.size())
#
#     input = torch.ones([2,2,5,5])
#     for i  in range(5):
#         for j in range(5):
#             input[:,:,i,j] = i * 5 + j
#
#     out = torch.pow(input,2)
#     # print(out)
