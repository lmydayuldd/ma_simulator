# import multi_sfm as sfm
import multi_cdt as sfm
import util
from pandas import DataFrame
from global_config import gc

def run(save_csv=False, save_anime=False):
    peds = []
    props = sfm.Props(repulsive_r=0.2)
    cols = ["STEP", "ID", "x", "y", "v_x", "v_y", "f_x", "f_y", "KIND"]
    dest_order1 = [0,2,3,4,5,6,1,8,-1]
    dest_order2 = [8,5,4,3,2,6,0,-1]
    mat = []
    flags = True
    step = 0
    # execute simulation
    while flags:
        if step%6 ==0 and step < 400:
            peds.append(sfm.Pedestrians(props, step, dest_order1,origin="entrance", kind=0))
        if step%6 ==3 and step < 400:
            peds.append(sfm.Pedestrians(props, step, dest_order2,origin="entrance", kind=1))

        print "step:{0},num:{1}".format(step, len(peds))
        step += 1
        peds = sfm.update(props, peds,timescale=1)
        if len(peds) == 0 or step > gc["iteration"]:
            flags = False
        util.record_data(step, peds, mat)
    frame = DataFrame(mat, columns=cols)
    # create animation
    if save_anime:
        anime_file = "{}_{}_{}.mp4" \
            .format(props.tau, props.V_to_ped, props.b_to_ped)
        util.anime(frame, anime_file, save_anime)
    if save_csv:
        file_name = "{}_{}_{}.csv" \
            .format(props.tau, props.V_to_ped, props.b_to_ped)
        frame.to_csv(file_name)
    return frame

if __name__ == '__main__':
    run(save_anime=True)