import numpy as np
from matplotlib import pyplot as plt

def LCVS(ex):
    if ex == "base":
        L,C,V,S = 60,14,7,10
    else:
        L,C,V,S = 60,16,8,11
    return L,C,V,S
def main(filepath, ex_name):
    plt.rcParams["font.size"] = 8
    fig1 = plt.figure()
    plt.title("Vertices Distance Average")

    for ex in ex_name:
        L,C,V,S = LCVS(ex)
        # fig = plt.figure()
        eval = open(filepath + ex + "/log-eval.txt", "r")
        loss = open(filepath + ex + "/log-loss.txt", "r")
        eval_val = np.array(list(map(float, eval.read().split())))
        plt.title("Angle Loss")
        plt.xlabel("Epoch", size=12)
        plt.ylabel("Evaluation", size=12)
        plt.xlim(-1000,80000)
        plt.ylim(0,100)

        eval_val = eval_val.reshape(len(eval_val)//C,C)
        plt.plot(eval_val[:,0], eval_val[:,V], label=ex + "_val")

        # train eval
        plt.plot(eval_val[:,0], eval_val[:,1], label=ex + "_loss")
        plt.legend()
        fig1.savefig(filepath + ex + "/val_plot.png", format="png", dpi=300)
    plt.close()

    fig2 = plt.figure()
    plt.title("Skeleton Joint (MPJPE)")
    for ex in ex_name:
        L,C,V,S = LCVS(ex)
        # fig = plt.figure()
        eval = open(filepath + ex + "/log-eval.txt", "r")
        loss = open(filepath + ex + "/log-loss.txt", "r")
        eval_val = np.array(list(map(float, eval.read().split())))
        plt.xlabel("Epoch", size=12)
        plt.ylabel("SKE evaluation (cm)", size=12)
        plt.xlim(-1000,80000)
        plt.ylim(0,200)

        eval_val = eval_val.reshape(len(eval_val)//C,C)
        plt.plot(eval_val[:,0], eval_val[:,S]*100, label=ex + "_val")

        # train eval
        plt.plot(eval_val[:,0], eval_val[:,4], label=ex + "_loss")
        plt.legend()
        fig2.savefig(filepath + ex + "/ske_val_plot.png", format="png", dpi=300)
    plt.close()

    fig3 = plt.figure()
    plt.title("Total Loss")
    for ex in ex_name:
        L,C,V,S = LCVS(ex)
        # fig = plt.figure()
        eval = open(filepath + ex + "/log-eval.txt", "r")
        loss = open(filepath + ex + "/log-loss.txt", "r")
        eval_val = np.array(list(map(float, eval.read().split())))
        plt.xlabel("Epoch", size=12)
        plt.ylabel("Total Loss", size=12)
        plt.xlim(-1000,80000)
        plt.ylim(0,100)

        eval_val = eval_val.reshape(len(eval_val)//C,C)
        eval_total = np.zeros((eval_val.shape[0]))
        for i in range(V,C):
            if C == 16 and i == 13:
                continue
            eval_total += eval_val[:,i]
        eval_total2 = np.zeros((eval_val.shape[0]))
        for i in range(1,V):
            if C == 16 and i == 6:
                continue
            eval_total2 += eval_val[:,i]

        plt.plot(eval_val[:,0], eval_total, label=ex + "_val")
        plt.plot(eval_val[:,0], eval_total2, label=ex + "_loss")

        min_pos = np.argmin(eval_total)
        plt.scatter(eval_val[min_pos,0],eval_total[min_pos],s=8,c="red")

        plt.legend()
        fig3.savefig(filepath + ex + "/total_val_plot.png", format="png", dpi=300)
    plt.close()


if __name__ == '__main__':
    filepath = "/home/jack/ドキュメント/002/mmWave_Stable/mmWave/1_python_codes/M3_deep_model/results/13-05_80000_128/"
    # ex_n = ["mmMesh_7", "mmMesh_7_model4_2", "mmMesh_7_model4_3_wd", "mmMesh_7_model4_3"]
    # ex_N = ["base", "base+D", "base+A", "base+DA"]

    ex_name = ["base"]

    # ex_n = ["mmMesh_base","mmMesh_model4_3"]
    # ex_N = ["base","base+AD",]
    # ex_name = ["mmMesh_7","mmMesh_7_model4_2",]
    main(filepath, ex_name)