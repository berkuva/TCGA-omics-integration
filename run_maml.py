import torch
import torch.nn as nn
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler
from utils import *
from maml_model import *
from platform import python_version
from torch import Tensor
import scipy.stats
from numpy.lib.recfunctions import unstructured_to_structured

# Python version
print(python_version())  # '3.7.13'
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print("Source cancer types: {}".format(source_tasks))
print("Target cancer types: {}".format(target_tasks))

# Load filtered and imputed transcriptomics, clinical, and proteomics datasets
PATH = "path/to/directory"
X_rna = pd.read_csv("{0}rna_imputed*.csv".format(PATH), sep='\t', index_col=0)
X_cli = pd.read_csv("{0}clinical_imputed*.csv".format(PATH), sep='\t', index_col=0)
X_pro = pd.read_csv("{0}proteomics_imputed*.csv".format(PATH), sep='\t', index_col=0)
X_rna.index = X_rna.index.astype(str)
X_cli.index = X_cli.index.astype(str)
X_pro.index = X_pro.index.astype(str)

# Gather censorship, duration, and cancer type information for survival analysis with cox-loss
clinical_df = build_clinical_dataframe('path-to-clinical-dataset-directory')
censorship = get_censorship_from_clinical(clinical_df)
duration = get_duration_from_clinical(clinical_df)
tumor_site = get_tumor_site_from_clinical(clinical_df)

# Integrate omics, contains patients not in source or target cancer types.
df_cox = pd.concat([X_cli, X_pro, X_rna], axis=1)
print("Shape of integrated dataset, patients by features: {0}".format(df_cox.shape))

# Further remove features with high Pearson correlation
df_cox = correlation(df_cox, 0.7)

# Get source data for source cancer types
target0_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 0)].sample(frac=1)
target5_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 5)].sample(frac=1)
target6_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 6)].sample(frac=1)
target2_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 2)].sample(frac=1)
target19_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 19)].sample(frac=1)
target13_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 13)].sample(frac=1)
target27_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 27)].sample(frac=1)
target29_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 29)].sample(frac=1)
target7_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 7)].sample(frac=1)
target8_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 8)].sample(frac=1)
target10_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 10)].sample(frac=1)
target12_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 12)].sample(frac=1)
target15_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 15)].sample(frac=1)
target21_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 21)].sample(frac=1)
target25_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 25)].sample(frac=1)

# Get source data for target cancer types
target4_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 4)].sample(frac=1)
target14_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 14)].sample(frac=1)
target17_df = df_cox.loc[get_pids_from_tumor_id(tumor_site, 17)].sample(frac=1)

# Split into support and query, with query set size of 30
n = 30
target0_df_query = target0_df.iloc[:n]
target0_df = target0_df[n:]
target5_df_query = target5_df.iloc[:n]
target5_df = target5_df[n:]
target6_df_query = target6_df.iloc[:n]
target6_df = target6_df[n:]
target2_df_query = target2_df.iloc[:n]
target2_df = target2_df[n:]
target19_df_query = target19_df.iloc[:n]
target19_df = target19_df[n:]
target13_df_query = target13_df.iloc[:n]
target13_df = target13_df[n:]
target29_df_query = target29_df.iloc[:n]
target29_df = target29_df[n:]
target7_df_query = target7_df.iloc[:n]
target7_df = target7_df[n:]
target8_df_query = target8_df.iloc[:n]
target8_df = target8_df[n:]
target10_df_query = target10_df.iloc[:n]
target10_df = target10_df[n:]
target12_df_query = target12_df.iloc[:n]
target12_df = target12_df[n:]
target15_df_query = target15_df.iloc[:n]
target15_df = target15_df[n:]
target21_df_query = target21_df.iloc[:n]
target21_df = target21_df[n:]
target25_df_query = target25_df.iloc[:n]
target25_df = target25_df[n:]
target4_df_query = target4_df.iloc[:n]
target14_df_query = target14_df.iloc[:n]
target17_df_query = target17_df.iloc[:n]
target4_df = target4_df[n:]
target14_df = target14_df[n:]
target17_df = target17_df[n:]


def get_data(task, query=False):
    """
    :param task: cancer type.
    :param query: True if desired data is query set; False if support.
    :return: data containing desired cancer type and support/query set.
    """
    if task == "ACC":
        if query:
            return target0_df_query
        return target0_df
    elif task == "GBM":
        if query:
            return target5_df_query
        return target5_df
    elif task == "BRCA":
        if query:
            return target6_df_query
        return target6_df
    elif task == "LGG":
        if query:
            return target2_df_query
        return target2_df
    elif task == "UCEC":
        if query:
            return target19_df_query
        return target19_df
    elif task == "OV":
        if query:
            return target13_df_query
        return target13_df
    elif task == "UCS":
        if query:
            return target29_df_query
        return target29_df
    elif task == "LUSC":
        if query:
            return target7_df_query
        return target7_df
    elif task == "CESC":
        if query:
            return target8_df_query
        return target8_df
    elif task == "COAD":
        if query:
            return target10_df_query
        return target10_df
    elif task == "ESCA":
        if query:
            return target12_df_query
        return target12_df
    elif task == "KIRP":
        if query:
            return target15_df_query
        return target15_df
    elif task == "PAAD":
        if query:
            return target21_df_query
        return target21_df
    elif task == "STAD":
        if query:
            return target25_df_query
        return target25_df
    elif task == "BLCA":
        if query:
            return target4_df_query
        return target4_df
    elif task == "HNSC":
        if query:
            return target14_df_query
        return target14_df
    elif task == "LUAD":
        if query:
            return target17_df_query
        return target17_df


class MAML():
    def __init__(self,
                 input_dim,
                 model,
                 inner_lr,
                 meta_lr,
                 K=1,
                 inner_steps=1,
                 tasks_per_meta_batch=1):

        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        self.meta_optimiser = torch.optim.SGD(self.weights, meta_lr)
        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # metrics
        self.print_every = 1
        self.concordance_index = []

    def get_duration(self, indices):
        return df_cox["duration"].iloc[indices].values

    def inner_loop(self, task, test_time=False):
        print(task)
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]

        # perform training on data sampled from source task support set
        df_cox_task = get_data(task, False)
        df_in = df_cox_task.drop(["duration", "censorship"], axis=1)
        X = torch.FloatTensor(df_in.values).to(device)

        for step in range(self.inner_steps):
            theta = self.model.parameterised(X, temp_weights)
            cox_loss = calculate_cox_loss(df_in=df_cox_task,
                                          theta=theta)

            # compute grad and update inner loop weights
            if test_time == False:
                grad = torch.autograd.grad(cox_loss, temp_weights)
                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # sample new data for meta-update and compute loss
        df_cox_task_query = get_data(task, True)

        duration = df_cox_task_query["duration"].values

        df_in = df_cox_task_query.drop(["duration", "censorship"], axis=1)

        X = torch.FloatTensor(df_in.values).to(device)

        theta = self.model.parameterised(X, temp_weights)
        cox_loss = calculate_cox_loss(df_in=df_cox_task_query,
                                      theta=theta)

        c_index = CIndex(theta, duration, df_cox_task_query["censorship"].values)

        if test_time:
            self.concordance_index.append(c_index)

        return cox_loss

    def main_loop_source(self, num_iterations):
        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):
            # compute meta loss
            meta_loss = 0

            for i in range(self.tasks_per_meta_batch):
                randnum = random.randint(0, len(source_tasks) - 1)
                task = source_tasks[randnum]
                loss = self.inner_loop(task)

                meta_loss += loss
            print(loss)

            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            self.model.zero_grad()
            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g

            self.meta_optimiser.step()

            # log metrics
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
            if iteration % self.print_every == 0:
                print("Avg C-index: ", round(np.mean(self.concordance_index), 2))
                print("!")
                epoch_loss = 0

    def main_loop_target(self, num_iterations, target_task, test_time=False):
        self.concordance_index = []
        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):
            meta_loss = 0

            for i in range(self.tasks_per_meta_batch):
                loss = self.inner_loop(target_task, test_time=True)
                meta_loss += loss

            # compute meta gradient of loss with respect to maml weights
            if test_time == False:
                meta_grads = torch.autograd.grad(meta_loss, self.weights)
                self.model.zero_grad()
                # assign meta gradient to weights and take optimisation step
                for w, g in zip(self.weights, meta_grads):
                    w.grad = g

                self.meta_optimiser.step()

            # log metrics
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
            if iteration % self.print_every == 0:  # self.print_every
                cindex = round(np.mean(self.concordance_index), 2)
                print("\nAvg C-index: ", cindex)
                print("@")
                self.concordance_index.append(cindex)

                epoch_loss = 0


meta_lr = 1e-4
inner_lr = 1e-4
input_dim = df_cox.shape[1] - 2
hidden_dim = input_dim // 5
output_dim = 1
model = MAMLModel(input_dim, hidden_dim, output_dim).to(device)

filepath = PATH
# filepath = PATH+"/new_maml_cox.pth"
# filepath = PATH+"/new_maml_cox_proteomics.pth"
# filepath = PATH+"/new_maml_cox_clinical.pth"
filepath = PATH + "/new_maml_cox_rna.pth"
# filepath = PATH+"/new_maml_cox_rna_clinical.pth"
# filepath = PATH+"/new_maml_cox_rna_pro.pth"
# filepath = PATH+"/new_maml_cox_clinical_proteomics.pth"


# Train on source tasks
if os.path.exists(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = MAMLModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    maml = MAML(input_dim, model, inner_lr=inner_lr, meta_lr=meta_lr)
    maml.meta_optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded saved model.")
else:
    print("No saved model found. Starting training.")
    maml = MAML(input_dim, model, inner_lr=inner_lr, meta_lr=meta_lr)
    maml.main_loop_source(num_iterations=100)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': maml.meta_optimiser.state_dict(),
    }, filepath)

# Fine-tune and evaluate on target tasks.
tt = "BLCA"  # HNSC; LUAD
for i in range(30):
    print(i)
    model.train()
    maml.main_loop_target(num_iterations=1, target_task=tt, test_time=False)  # finetune
    model.eval()
    maml.main_loop_target(num_iterations=1, target_task=tt, test_time=True)  # evaluation
    torch_compute_confidence_interval(torch.FloatTensor(maml.concordance_index))

filepath_ = "path-to-save-model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': maml.meta_optimiser.state_dict(),
    }, filepath_)
