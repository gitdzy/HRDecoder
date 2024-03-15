SIGMOID_THRESH = 0.5

from tkinter import Label
import numpy as np
from sklearn.metrics import auc,roc_auc_score,mean_absolute_error,roc_curve

np.seterr(invalid='ignore')

"""
results:
gt:
[ (2848, 4288),  ...]

results {0, 1, 2, 3 ,4}
gt {0, 1, 2, 3, 4}

0 stands for background
"""


# softmax
def softmax_confused_matrix(pred_label, label, num_classes):
    tp = pred_label[pred_label == label]

    area_p, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_tp, _ = np.histogram(tp, bins=np.arange(num_classes + 1))
    area_gt, _ = np.histogram(label, bins=np.arange(num_classes + 1))

    area_fn = area_gt - area_tp

    return area_p, area_tp, area_fn


def softmax_metrics(results, gt_seg_maps, num_classes):
    """
    :param results:  {0, 1, 2, 3 ,4}
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs

    total_p = np.zeros((num_classes,), dtype=np.float64)
    total_tp = np.zeros((num_classes,), dtype=np.float64)
    total_fn = np.zeros((num_classes,), dtype=np.float64)
    maupr = np.zeros((num_classes,), dtype=np.float64)

    #==========================================
    mae=np.zeros((num_classes,),dtype=np.float64)
    mauroc=np.zeros((num_classes,),dtype=np.float64)
    #==========================================

    for i in range(num_imgs):
        if isinstance(results[i], tuple):
            result = results[i][0]
            result = np.argmax(result, axis=0)
        else:
            result = results[i]

        p, tp, fn = softmax_confused_matrix(result, gt_seg_maps[i], num_classes)
        total_p += p
        total_tp += tp
        total_fn += fn

#==========================================
    return total_p, total_tp, total_fn, maupr,mae,mauroc
#==========================================


# sigmoid
def sigmoid_confused_matrix(pred_logit, raw_label, num_classes, thresh):
    # 这个函数也是每次都针对一张图片计算的，然后根据4种病灶计算这些值的
    assert pred_logit.shape[0] == num_classes - 1

    class_p = np.zeros((num_classes,), dtype=np.float64)
    class_tp = np.zeros((num_classes,), dtype=np.float64)
    class_fn = np.zeros((num_classes,), dtype=np.float64)
    #==========================================
    class_tn = np.zeros((num_classes,),dtype=np.float64)
    #class_ae = np.zeros((num_classes,),dtype=np.float64)  #计算mae
    #class_auroc = np.zeros((num_classes,),dtype=np.float64) #计算auroc
    # 既然是一张图片，就应该是（4,1）的维度
    #==========================================
    # 这里是针对每一类都计算

    for i in range(1, num_classes):
        pred = pred_logit[i - 1] > thresh
        label = raw_label == i      #把label化成二值图的形式，对应每种病灶的label
        class_tp[i] = np.sum(label & pred)
        class_p[i] = np.sum(pred)
        class_fn[i] = np.sum(label) - class_tp[i]
        class_tn[i] = np.sum((label | pred)==0)
        #class_ae[i] = mean_absolute_error(label,pred_logit[i-1])
        #class_auroc[i] = roc_auc_score(label,pred_logit[i-1])
    
    #print(class_ae)
    return class_p, class_tp, class_fn, class_tn


def sigmoid_ae(results,gt_seg_maps,num_classes):
    '''
    results: a list of tuple composed of (list,bool,bool), 
    in which the first para means the prediction map,second represents the value of use_sigmoid,the third for compute_aupr

    gt_seg_maps:a list of  the label of feature maps(each lesion using an integer to refer to)

    '''

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs     # 这里早tm就已经把长度给我总结好了，谢谢

    total_ae = np.zeros((num_classes,),dtype=np.float64)

    for i in range(num_imgs):
        if isinstance(results[i],tuple):      # 这里是确保拿到的是图的预测结果而不是带有两个true值的元组
            pred_logit=results[i][0]
        else:
            pred_logit=results[i]

        class_ae = np.zeros((num_classes,),dtype=np.float64)
        
        for j in range(1,num_classes):
            label = gt_seg_maps[i]==j   #把label化为二值图
            class_ae[j] = mean_absolute_error(label,pred_logit[j-1])
        
        total_ae += class_ae

    return total_ae/num_imgs


def sigmoid_auroc(results,gt_seg_maps,num_classes):
    '''
    results: a list of tuple composed of (list,bool,bool), 
    in which the first para means the prediction map,second represents the value of use_sigmoid,the third for compute_aupr

    gt_seg_maps:a list of  the label of feature maps(each lesion using an integer to refer to)
    '''
    # 思考了一下，感觉roc应该只能在这里计算，因为roc本来就应该是针对一组数据来计算的，所以应该放在这个函数中计算
    # 之前是放在了sigmoid_confused_matrix中计算了，后来想想还是解耦比较好
    # 因为一开始计算出来的roc有问题，这里用roc_curve和roc_auc_score两种都试一下
    # 把ae和roc都单独拿出来算，所以最后的结果也需要先除以图片的数量

    num_imgs= len(results)
    assert len(gt_seg_maps) == num_imgs

    total_auroc = np.zeros((num_classes,),dtype=np.float64)
    lesion_nums = np.zeros((num_classes,),dtype=np.int16)
    lesion_nums += num_imgs             #为了对每一种病灶都计数
    '''

    for i in range(num_imgs):
        if isinstance(results[i],tuple):
            pred_logit=results[i][0]
        else:
            pred_logit=results[i]

        # print('pred_logit',pred_logit)
        class_auroc = np.zeros((num_classes,),dtype=np.float64)

        for j in range(1,num_classes):
            label = gt_seg_maps[i]==j
            # 第一种计算方式，用自带的roc_auc_score函数
            # roc_auc_score函数要求y_true一定要有两种label（比如0和1，就算只有0也会报错的，所以需要自己处理一下数据或用try-except跳过）
            try:         #这里是确保gt中至少有一个像素是true，这样说明肯定有这个病灶,顺便说一下，这个roc_auc_score只能作用于一维向量
                class_auroc[j] = roc_auc_score(label.flatten(),pred_logit[j-1].flatten())
                # 感觉有问题，这样根本算不了
                #fpr, tpr,_=roc_curve(label.flatten(),pred_logit[j-1].flatten(),pos_label=True)
                #auroc = auc(fpr,tpr)
            except:
                #如果全是0就跳过该病灶gt，（应该不会出现全是1的图片所以不考虑）
                # 不过这样就听麻烦的，因为需要对每种病灶都去记个数，反向计数，如果全是0表示没有这个病灶
                #应该不至于一种病灶连一个gt都没有吧
                lesion_nums[j]-=1       

        total_auroc+=class_auroc
        # print('class_auroc',class_auroc)
    '''
    return total_auroc/lesion_nums  



# sigmoid
#==========================================
# 无奈，只能先把这两句给去掉compute_mae=False,compute_roc=False
def sigmoid_metrics(results, gt_seg_maps, num_classes, compute_aupr=False):
#==========================================
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs     # 这里早tm就已经把长度给我总结好了，谢谢

    if compute_aupr:        # 这里的采样频率就是逼近PR/ROC曲线的的采样（一般来说是采样越多逼近效果越好的）
        # threshs = np.linspace(0, 1, 33)  # 0.03125
        threshs = np.linspace(0, 1, 11)  # 0.1
        #threshs = np.linspace(0, 1, 21)  # 0.05
    else:
        threshs = [SIGMOID_THRESH]

    total_p_list = []
    total_tp_list = []
    total_fn_list = []

    total_tn_list=[]

    # 这里全是为了通过选择阈值的方式来逼近曲线，所以如果是计算MAE的话没必要
    for thresh in threshs:
        total_p = np.zeros((num_classes,), dtype=np.float64)
        total_tp = np.zeros((num_classes,), dtype=np.float64)
        total_fn = np.zeros((num_classes,), dtype=np.float64)
        total_tn = np.zeros((num_classes,), dtype=np.float64)

        for i in range(num_imgs):   #每次单独送入一张图片计算
            if isinstance(results[i], tuple):
                result = results[i][0]
            else:
                result = results[i]

            p, tp, fn,tn = sigmoid_confused_matrix(result, gt_seg_maps[i], num_classes, thresh)
            total_p += p
            total_tp += tp
            total_fn += fn
            total_tn += tn

        total_p_list.append(total_p)
        total_tp_list.append(total_tp)
        total_fn_list.append(total_fn)
        total_tn_list.append(total_tn)

    if len(threshs) > 1:        #选择阈值是0.5时的图片,如果是MAE的话就应该是在thresh=0的时候的取值
        index = int(np.argmax(threshs == SIGMOID_THRESH))
    else:
        index = 0

    total_p = total_p_list[index]
    total_tp = total_tp_list[index]
    total_fn = total_fn_list[index]
    total_tn = total_tn_list[index]

    # 后来重新看了一下自己写的代码，应该没有问题，不管index是什么都一样，因为我用的是pred_logit和label在计算，没有用到阈值，所以无所谓

    #print(total_ae)
    mae = np.zeros((num_classes,),dtype=np.float64)
    #print(mae)

    maupr = np.zeros((num_classes,), dtype=np.float64)
    #==========================================
    # mae = np.zeros((num_classes,), dtype=np.float64)
    mauroc = np.zeros((num_classes,), dtype=np.float64)
    #==========================================
    total_p_list = np.stack(total_p_list)
    total_tp_list = np.stack(total_tp_list)
    total_fn_list = np.stack(total_fn_list)
    total_tn_list = np.stack(total_tn_list)

    ppv_list = np.nan_to_num(total_tp_list / total_p_list, nan=1)
    s_list = np.nan_to_num(total_tp_list / (total_tp_list + total_fn_list), nan=0)

    # tpr_list = s_list 因为tpr（灵敏度其实就是召回率）
    # fpr_list = np.nan_to_num((total_p_list-total_tp_list)/(total_p_list-total_tp_list+total_fn_list), nan=0)
    # print(ppv_list)
    # print(s_list)

    if compute_aupr:
        for i in range(1, len(maupr)):
            x = s_list[:, i]
            y = ppv_list[:, i]
            maupr[i] = auc(x, y)

    # print(maupr.shape) (5,)很正常，因为
    #==========================================
    # 思考了一下，这里从1开始标号应该是因为bg不用考虑

    # print(results[0][0],results[0][0].shape) (4,2848,3414)
    # print(gt_seg_maps[0],gt_seg_maps[0].shape) (2848,3414)，gt_map里面其实是存储了好几种病灶，用不同的数字来表示不同的病灶，所以没有维度4
    # mae要写成4维的，因为要考虑4中病灶
    # 经过一番理解，我觉得这个mae不是和aupr和auroc一样计算的，而是取阈值维0.5时候的预测结果来计算的
    '''
    for i in range(1,len(mae)):
        # gt_map里面其实是存储了好几种病灶，用不同的数字来表示不同的病灶
        label = gt_seg_maps[i]==i    #这里把gt图转化为二值图
        mae[i]=np.linalg.norm(label-pred)/num_imgs      #麻烦下次写代码快一点，就一行代码，会不会写啊
    '''
    
    #sklearn.metrics有roc_auc_score()函数，但是这个是只能用于二分类的，没有auc通用，还是推荐用auc吧
    # 后来感觉这样写结果就不对，因为计算曲线下的面颊应该是要先把pred和label按照置信度排序后再计算
    # 关于这里的计算，因为真实曲线是很难画出来的，所以其实是多次选择阈值然后去逼近的，
    # 体现在代码里面其实就是从94行到129行在干的事情
    '''

    for i in range(1,len(mauroc)):      
        x= fpr_list[:,i]
        y= s_list[:,i]
        mauroc[i]=auc(x,y)
    '''

    mae = sigmoid_ae(results,gt_seg_maps,num_classes)
    mauroc = sigmoid_auroc(results,gt_seg_maps, num_classes)

    return total_p, total_tp, total_fn, maupr, mae, mauroc
    #==========================================

# 然后要注意了，既然改了这里，那么my_custom.py文件也要改，因为那里才是输出并调用我们测试用的代码的接口
# 记得做好版本的备份，别tm到时候回不去了，烧包
# 给我搞清楚这个result是哪里来的谢谢
def metrics(results, gt_seg_maps, num_classes, ignore_index=None, nan_to_num=None):
    """
    :param results: feature map after sigmoid of softmax
    """

    compute_aupr = False
    use_sigmoid = False

    #==========================================
    #compute_mae=False
    #compute_roc=False
    #==========================================
    # 这里的result中会传回来一个是计算结果，另外两个是use_sigmoid和compute_aupr的bool值
    if isinstance(results[0], tuple):
        _, use_sigmoid, compute_aupr = results[0]

    if not use_sigmoid:
        total_p, total_tp, total_fn, maupr, mae, mauroc = softmax_metrics(
            results, gt_seg_maps, num_classes)
    else:
        total_p, total_tp, total_fn, maupr, mae, mauroc = sigmoid_metrics(
            results, gt_seg_maps, num_classes, compute_aupr)

    ppv = total_tp / total_p                # ppv中total_p就是tp+fp（就是pred）
    s = total_tp / (total_tp + total_fn)        #所以PPV就是精确率，s就是召回率
    f1 = 2 * total_tp / (total_p + total_tp + total_fn)     #total_p=total_tp+total_fp，所以p+tp+fn=2*tp+fp+fn，公式没问题
    # f1 = (s * ppv * 2) / (s + ppv)
    iou = total_tp / (total_p + total_fn)

    if nan_to_num is not None:
        return np.nan_to_num(iou, nan=nan_to_num), \
               np.nan_to_num(f1, nan=nan_to_num), \
               np.nan_to_num(ppv, nan=nan_to_num), \
               np.nan_to_num(s, nan=nan_to_num), \
               np.nan_to_num(maupr, nan=nan_to_num), \
               np.nan_to_num(mae,nan=nan_to_num), \
               np.nan_to_num(mauroc,nan=nan_to_num)
    else:
        return iou, f1, ppv, s, maupr, mae, mauroc


if __name__ == '__main__':
    shape = [3, 4]
    num_classes = 4  # include background
    num = 2
    use_sigmoid = True
    aupr = False

    pred = [(np.random.random([num_classes, shape[0], shape[1]]), use_sigmoid, aupr) for i in range(num)]
    label = [np.random.randint(0, num_classes + 1, shape) for i in range(num)]

    res = metrics(pred, label, num_classes + 1)
    for i in res: print(i)

    # x = np.random.random(10)
    # y = np.random.random(10)
    # print(auc(x, y))
    # print(roc_auc_score(x, y))
