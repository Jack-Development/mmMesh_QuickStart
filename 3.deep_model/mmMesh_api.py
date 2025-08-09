from infer_single import mmwave


class mmMeshAPI:
    def run_inference(self, model_path, pc_size, input_path):
        m = mmwave(model_path, pc_size)
        q, t, v, s, l, b, g, pc = m.infer(input_path)

        # Dictionary:
        # q: 9x3x3 local joint rotation matrices
        # t: global translation vectors
        # v: SMPL Mesh vertices
        # s: SMPL joint positions
        # l: 2-D Global "location" predicted by the Global RNN
        # b: Shape betas
        # g: Gender logit
        # pc: Raw point cloud data

        return {
            "skeleton": s,
            "pose": q,
            "trans": t,
        }
