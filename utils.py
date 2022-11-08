import torch
import numpy as np
from torch import Tensor
import scipy.stats
import glob
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

source_tasks = ["ACC", "GBM", "BRCA", "LGG", "UCEC", "OV", "UCS", "LUSC",
                "CESC", "COAD", "ESCA", "KIRP", "PAAD", "STAD"]
target_tasks = ["BLCA", "HNSC", "LUAD"]


# stackoverflow.com/questions/29294983
def correlation(dataset, threshold):
    """
    :param dataset: A DataFrame.
    :param threshold: By which features at or above this value of Pearson correlation will be removed.
    :return: Filtered DataFrame.
    """
    col_corr = set()
    corr_matrix = dataset.corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]

    return dataset


# Build a dataframe with TCGA clinical data for all cancer types for collecting
# censorship, duration, and patients' information
def build_clinical_dataframe(clinical_dir):
    """
    :param clinical_dir: path to directory containing the TCGA clinical datasets for cancer types.
    :return: A merged DataFrame.
    """
    clinical = glob.glob('{}/*.txt'.format(clinical_dir))

    clinical_df = pd.DataFrame()

    for i in range(len(clinical)):
        tumor_type = clinical[i].split("/")[-1].split("-")[0]
        if tumor_type in source_tasks or tumor_type in target_tasks:
            df = pd.read_csv(clinical[i], sep='\t', low_memory=False)

            pids = df.columns[1:]
            pids = [pid.split("-")[-1] for pid in pids]

            columns = df["tcga_participant_barcode"]
            df = df.T
            df = df.iloc[1:]
            df.columns = columns
            df.index = pids
            clinical_df = clinical_df.append(df)
    clinical_df = clinical_df.drop(["Unnamed: 549", "all_p", "all_q"])

    return clinical_df


def get_censorship_from_clinical(clinical_df):
    """
    :param clinical_df: A merged DataFrame containing clinical dataset.
    :return: censorship data for patients in clinical_df.
    """
    censored = clinical_df["CLI_days_to_death"].isna()
    censorship = censored.replace([False, True], [0, 1])
    return censorship


def get_duration_from_clinical(clinical_df):
    """
    :param clinical_df: A merged DataFrame containing clinical dataset.
    :return: duration for patients in clinical_df.
    """
    days_to_death = clinical_df["CLI_days_to_death"].loc[not clinical_df["CLI_days_to_death"].isna()]
    days_to_last_followup = clinical_df["CLI_days_to_last_followup"].loc[
        not clinical_df["CLI_days_to_last_followup"].isna()]
    duration = days_to_death.append(days_to_last_followup)
    return duration


def get_tumor_site_from_clinical(clinical_df):
    """
    :param clinical_df: A merged DataFrame containing clinical dataset.
    :return: tumor_sites for patients in cl inical_df.
    """
    return clinical_df["CLI_tumor_tissue_site"]


def convert_cancer_types_string_to_int(tumor_site):
    """
    :param tumor_site: tumor sites in string
    :return: tumor sites in int
    """
    tumor_site = tumor_site.replace({'lusc': 7,
                                     'ovary': 13,
                                     'brain': 5,
                                     'luad': 17,
                                     'omentum': 13,
                                     'colon': 10,
                                     'kidney': 15,
                                     'head and neck': 14,
                                     'stomach': 25,
                                     'central nervous system': 2,
                                     'pancreas': 21,
                                     'breast': 6,
                                     'endometrial': 19,
                                     'bladder': 4,
                                     'cervical': 8,
                                     'thyroid': 28,
                                     'esophagus': 12,
                                     'uterus': 29,
                                     'adrenal': 0,
                                     'adrenal gland': 1,
                                     'extra-adrenal site': 1,
                                     'testes': 26,
                                     'bile duct': 3,
                                     'choroid': 9,
                                     'thymus': 27,
                                     'anterior mediastinum': 27})
    return tumor_site


def get_pids_from_tumor_id(tumor_site, tumor_id):
    return tumor_site[tumor_site["CLI_tumor_tissue_site"] == tumor_id].index


# https://github.com/gevaertlab/metalearning_survival
def calculate_cox_loss(df_in, theta):
    """
    :param df_in: dataframe of uncensored patients on whom cox loss is calculated, needed for getting the duration.
    :param theta: meta-learning model output.
    :return: cox hazard loss.
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

    return loss


# https://github.com/gevaertlab/metalearning_survival
def CIndex(pred, ytime_test, ystatus_test):
    """
    :param pred: meta-learning output.
    :param ytime_test: duration for uncensored patients.
    :param ystatus_test: patients' censorship.
    :return: concordance index
    """
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

    return concord / total


# pytorch discussion 139398
def torch_compute_confidence_interval(data: Tensor, confidence: float = 0.95):
    """
    :param data: tensor data on which confidence intervals will be calculated.
    :param confidence: confidence level.
    :return: mean and confidence interval.
    """
    n = len(data)
    mean: Tensor = data.mean()
    se: Tensor = data.std(unbiased=True) / (n ** 0.5)
    t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2., n - 1))
    ci = t_p * se
    return mean, ci
