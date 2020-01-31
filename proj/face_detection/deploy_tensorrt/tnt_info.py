import api_dirs
import os
import numpy as np

tnt_name_tup = ("Cong Pha", "Tan Tai",  # admins
                "Me", "Ba", "Chi",  # roomlord
                "Pham Gia Hao", "Truong Bao Tuyen",  # room1 
                "Nguyen Tan Phuc", "Tran Thi Tuyet Hong",  # room2
                "Tran Huu Nghia", "Bui Thi Thu Ha",  # room3
                "Huynh Thi My Duyen",  # room4
                "Nguyen Huy", "Nguyen Thi Truc Van",  # room5
                "Huynh Kim Lang", "Huynh Tay Nam",  # room6
                "Hao", "Nguyen Thi Bich Chau", "Khang",  # room7
                "Nguyen Ngoc Anh", "Nguyen Thi Ngoc Giau", "Nguyen Thi Ngan", "Nguyen Van Duoc",  # room8
                #"room9",  # room9
                "Nguyen Quoc Mai", #"Nguyen Thi Nguyet Nga",  # room10
                "Huynh Thanh Binh", "Yen")  # room11


def get_smpl_encs(candidate_id, current_time):
    """

    :param candidate_id: 0, 1, 2 ...
    :param current_time: 'd' or 'n'
    :return: a python list of that candidate's sample encodings
    """
    candidate_full_name = tnt_name_tup[candidate_id]
    print(f"[   info] getting smpl embs for {candidate_full_name} ... ")

    tnt_smpl_embs_dir = api_dirs.tnt_smpl_embs_dir
    for file in os.listdir(tnt_smpl_embs_dir):
        if file.startswith(str(candidate_id) + f"_{current_time}") and file.endswith(".npy"):
            candidate_smpl_encs_list = np.load(os.path.join(tnt_smpl_embs_dir, file))
            print("[   info] file used: {}".format(file))

    return list(candidate_smpl_encs_list)
