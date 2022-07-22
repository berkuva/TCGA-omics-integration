import torch
import numpy as np
from torch import Tensor
import scipy.stats

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


def get_source_tasks():
    # 8: CESC; 10: COAD; 12: ESCA; 15: KIRP; 21: PAAD; 25: STAD
    return [8, 10, 12, 15, 21, 25]


def get_target_tasks():
    # 4: BLCA; 14: HNSC; 17: LUAD/LUSC
    #     return [4]
    #     return [14]
    return [17]


def calculate_cox_loss(df_in, theta):
    """
    df_in: dataframe of uncensored patients on whom cox loss is calculated, needed for getting the duration
    features: patient features from neural network (# patients with tasks, # output_dim)
    """
    observed = df_in["censorship"].values
    observed = torch.FloatTensor(observed).to(device)

    df_in = df_in.reset_index()
    exp_theta = torch.exp(theta)
    exp_theta = torch.reshape(exp_theta, [exp_theta.shape[0]])
    theta = torch.reshape(theta, [theta.shape[0]])
    R_matrix_batch = np.zeros([exp_theta.shape[0], exp_theta.shape[0]], dtype=int)

    for i, row1 in df_in.iterrows():
        for j, row2 in df_in.iterrows():
            time_1 = row1["duration"]
            time_2 = row2["duration"]
            R_matrix_batch[i, j] = time_2 >= time_1

    R_matrix_batch = torch.FloatTensor(R_matrix_batch).to(device)

    loss = -(torch.sum(torch.mul(torch.sum(theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))),
                                 observed)) / len(observed))

    #     print(loss.item())

    return loss


# https://github.com/gevaertlab/metalearning_survival
def CIndex(pred, ytime_test, ystatus_test):
    concord = 0.
    total = 0.
    N_test = ystatus_test.shape[0]
    ystatus_test = np.asarray(ystatus_test, dtype=bool)
    theta = pred
    for i in range(N_test):
        # 0 is for uncensored
        if ystatus_test[i] == 0:
            for j in range(N_test):
                if ytime_test[j] >= ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        concord = concord + 1
                    elif theta[j] == theta[i]:
                        concord = concord + 0.5

    return (concord / total)


# https://discuss.pytorch.org/t/what-is-the-proper-way-to-compute-95-confidence-intervals-with-pytorch-for-classification-and-regression/139398/2
def torch_compute_confidence_interval(data: Tensor, confidence: float = 0.95):
    n = len(data)
    mean: Tensor = data.mean()
    se: Tensor = data.std(unbiased=True) / (n ** 0.5)
    t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2., n - 1))
    ci = t_p * se
    return mean, ci
