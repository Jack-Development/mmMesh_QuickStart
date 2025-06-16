from smpl_models.smpl_utils_extend import SMPL


class SMPLWrapper:
    def __init__(self):
        self.male_smpl = SMPL("m")
        self.female_smpl = SMPL("f")
