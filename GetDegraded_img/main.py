import utils.utils_train as ut
from GetDegraded_img.data_load import Degrader
from tqdm import tqdm



if __name__ == '__main__':

    target_root = ""
    gt_root = ""

    gt_list = []
    ut.listdir(gt_root,gt_list)

    degrader = Degrader()
    for i,file in tqdm(enumerate(gt_list)):
        degrader.degrade_single_img(target_root,file)



