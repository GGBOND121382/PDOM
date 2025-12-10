import os
import adv_setting.loop_DOCO_LR
import adv_setting.loop_PDOCO_LR
import adv_setting.loop_PDOMO_LR
import iid_setting.loop_DMA_LR
import iid_setting.loop_DOCO_nc_LR
import iid_setting.loop_PDOMS_LR
import iid_setting.loop_PFTAL_LR
import iid_setting.loop_PTOFW_LR


if __name__ == '__main__':

    os.chdir('iid_setting')
    # iid_setting.loop_DMA_LR.run()
    # iid_setting.loop_DOCO_nc_LR.run()
    # iid_setting.loop_PDOMS_LR.run()
    # iid_setting.loop_PFTAL_LR.run()
    # iid_setting.loop_PTOFW_LR.run()

    os.chdir('../adv_setting')
    adv_setting.loop_DOCO_LR.run()
    adv_setting.loop_PDOCO_LR.run()
    adv_setting.loop_PDOMO_LR.run()

