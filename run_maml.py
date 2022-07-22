import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import tqdm as tqdm
import json
from utils import *
from maml_model import *

PATH = ""

# torch.manual_seed(2)
# np.random.seed(3)
# random.seed(4)

X = pd.read_csv(PATH+"X.csv")
X_rna = pd.read_csv(PATH+"X_RNA.csv")
X_cli = pd.read_csv(PATH+"X_clinical.csv")
X_pro = pd.read_csv(PATH+"X_proteomics.csv")
censorship = pd.read_csv(PATH+"censorship.csv")['0'].values
times = pd.read_csv(PATH+"times.csv")['0'].values
tumor_site = pd.read_csv(PATH+"tumor_site.csv")['0'].values

with open(PATH + "patient2event_time.txt", 'r') as j:
    patient2event_time = json.loads(j.read())

X = torch.cat((torch.FloatTensor(X_rna.values), torch.FloatTensor(X_cli.values)), dim=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

aggregate_X_corr = correlation(X, 0.7)
X = X[list(aggregate_X_corr)]

X.index = list(patient2event_time.keys())
X['duration'] = times
X['censorship'] = censorship
df_cox = X

source_tasks = get_source_tasks()
target_tasks = get_target_tasks()

target4_df = df_cox.iloc[np.where(tumor_site == 4)]
target14_df = df_cox.iloc[np.where(tumor_site == 14)]
target17_df = df_cox.iloc[np.where(tumor_site == 17)]

target4_df = target4_df.sample(frac=1)
target14_df = target14_df.sample(frac=1)
target17_df = target17_df.sample(frac=1)

n = 30
target4_df_test = target4_df.iloc[:n]
target14_df_test = target14_df.iloc[:n]
target17_df_test = target17_df.iloc[:n]
target4_df = target4_df[n:]
target14_df = target14_df[n:]
target17_df = target17_df[n:]


def get_data(task, test=False):
    if task == 4 or task == 14 or task == 17:
        if task == 4:
            if test:
                return target4_df_test
            return target4_df
        elif task == 14:
            if test:
                return target14_df_test
            return target14_df
        else:
            if test:
                return target17_df_test
            return target17_df
    else:
        index = np.where(tumor_site==task)
        return df_cox.iloc[index]


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
        self.print_every = 10
        self.brier_scores = []
        self.concordance_index = []

    def get_duration(self, indices):
        return df_cox["duration"].iloc[indices].values

    def inner_loop(self, task, target_query=False, limit=False):
        print(task)
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]

        # perform training on data sampled from source task support set
        df_cox_task = get_data(task, test=False)
        df_in = df_cox_task.drop(["duration", "censorship"], axis=1)
        X = torch.FloatTensor(df_in.values).to(device)

        for step in range(self.inner_steps):
            theta = self.model.parameterised(X, temp_weights)
            cox_loss = calculate_cox_loss(df_in=df_cox_task,
                                          theta=theta)

            # compute grad and update inner loop weights
            if target_query == False:
                grad = torch.autograd.grad(cox_loss, temp_weights)
                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # sample new data for meta-update and compute loss
        df_cox_task_query = get_data(task, test=target_query)

        duration = df_cox_task_query["duration"].values

        df_in = df_cox_task_query.drop(["duration", "censorship"], axis=1)

        X = torch.FloatTensor(df_in.values).to(device)

        theta = self.model.parameterised(X, temp_weights)
        #         print(theta)
        cox_loss = calculate_cox_loss(df_in=df_cox_task_query,
                                      theta=theta)

        c_index = CIndex(theta, duration, df_cox_task_query["censorship"].values)
        #         c_index = CIndex(duration, theta.detach().numpy(), 1-df_cox_task_query["censorship"].values)

        self.concordance_index.append(c_index)

        return cox_loss

    def main_loop_source(self, num_iterations):
        epoch_loss = 0

        for iteration in tqdm.tqdm(range(1, num_iterations + 1)):
            # compute meta loss
            meta_loss = 0

            for i in range(self.tasks_per_meta_batch):
                randnum = random.randint(0, len(source_tasks) - 1)
                task = source_tasks[randnum]
                loss = self.inner_loop(task, target_query=False, limit=False)

                meta_loss += loss

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

    def main_loop_target(self, num_iterations, test_time=False):
        self.concordance_index = []
        epoch_loss = 0

        for iteration in tqdm.tqdm(range(1, num_iterations + 1)):
            # compute meta loss
            meta_loss = 0

            for i in range(self.tasks_per_meta_batch):
                randnum = random.randint(0, len(target_tasks) - 1)
                task = target_tasks[randnum]
                loss = self.inner_loop(task, target_query=test_time, limit=False)
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
input_dim = df_cox.shape[1]-2
hidden_dim = input_dim//5
output_dim = 1
model = MAMLModel(input_dim, hidden_dim, output_dim).to(device)

import os
filepath=PATH
# filepath = PATH+"/maml_cox.pth"
# filepath = PATH+"maml_cox_full_new_reduced.pth"
# filepath = PATH+"maml_cox_full_new_reduced_RNA.pth"
# filepath = PATH+"maml_cox_full_new_reduced_clinical.pth"
# filepath = PATH+"maml_cox_full_new_reduced_proteomics.pth"
filepath = PATH+"maml_cox_full_new_reduced_X_rna_clinical.pth"
# filepath = PATH+"maml_cox_full_new_reduced_X_rna_pro.pth"

print(filepath)



if os.path.exists(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = MAMLModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    maml = MAML(input_dim, model, inner_lr=inner_lr, meta_lr=meta_lr)
    maml.meta_optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded saved model.")
else:
    print("No saved model found. Starting training.")
    maml = MAML(input_dim, model, inner_lr=inner_lr, meta_lr=meta_lr)
    maml.main_loop_source(num_iterations=50)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': maml.meta_optimiser.state_dict(),
            }, filepath)
    print("Model saved.")



# Target finetune
# filepath_ = PATH + "{}maml_cox_full_new_reduced.pth".format(target_tasks)
# filepath_ = PATH + "{}maml_cox_full_new_reduced_RNA.pth".format(target_tasks)
filepath_ = PATH + "{}maml_cox_full_new_reduced_clinical.pth".format(target_tasks)
# filepath_ = PATH + "{}maml_cox_full_new_reduced_proteomics.pth".format(target_tasks)
# filepath_ = PATH + "{}maml_cox_full_new_reduced_X_rna_clinical.pth".format(target_tasks)
# filepath_ = PATH + "{}maml_cox_full_new_reduced_pro_clinical.pth".format(target_tasks)
# filepath_ = PATH + "{}maml_cox_full_new_reduced_rna_pro.pth".format(target_tasks)

for i in range(5):
    model = MAMLModel(input_dim, hidden_dim, output_dim).to(device)
    maml.main_loop_target(num_iterations=10, test_time=False) # finetune
    model.eval()
    print("---")
    maml.main_loop_target(num_iterations=30, test_time=True) # evaluation
    torch_compute_confidence_interval(torch.FloatTensor(maml.concordance_index))


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': maml.meta_optimiser.state_dict(),
    }, filepath_)


