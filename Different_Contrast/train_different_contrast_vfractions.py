import sys
import os
import time
from tqdm import trange
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from optimizer.soap import SOAP

torch.set_default_dtype(torch.float64)
print(torch.get_default_dtype())

path_soap = '/home/bmederos/pinns'
sys.path.append(path_soap)
save_dir = '/home/bmederos/pinns/save'

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
DEVICE = torch.device('cuda')

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.determinist = True
torch.backends.cudnn.benchmark = False

LAYERS = [1,100,100,100,1]
LOG_STEP = 5
date = '2026_02_04'

class FCN(nn.Module):

    def __init__(self, layers, device, activation = nn.Tanh()):

        super().__init__()

        self.device = device
        self.layers = layers
        self.activation = activation

        self.linear = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])

        for i in range(len(self.layers)-1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain = 1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, X):
        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X)
        a = X

        for i in range(len(self.layers)-2):
            z = self.linear[i](a)
            a = self.activation(z)

        NN_pred = self.linear[-1](a)
        return NN_pred
    
class PINN(nn.Module):

    def __init__(self, y_collocation_mat1 = None, y_collocation_mat2 = None, y_interface = None, 
                 param = None, layers = None, kappa = None, device = None, prob_name = None):

        super().__init__()

        self.prob_name = prob_name
        if y_interface != None:
           self.vf = y_interface[0,0]
        self.device = device

        if y_collocation_mat1 != None:
           self.y_mat1 = y_collocation_mat1
           #self.y_mat1 = torch.tensor(y_collocation_mat1, requires_grad= True).double().to(self.device)
           self.null_mat1 = torch.zeros(self.y_mat1.shape, dtype = torch.float64, requires_grad = False).to(self.device)

        if y_collocation_mat2 != None:
           self.y_mat2 = y_collocation_mat2
           #self.y_mat2 = torch.tensor(y_collocation_mat2, requires_grad= True).double().to(self.device)
           self.null_mat2 = torch.zeros(self.y_mat2.shape, dtype = torch.float64, requires_grad = False).to(self.device)

        if y_interface != None:
           self.y_inter= torch.as_tensor(y_interface, dtype=torch.float64, device=self.device).requires_grad_(True)
           self.null_inter = torch.zeros(self.y_inter.shape, dtype = torch.float64, requires_grad=False).to(self.device)

        #boundary points
        self.y_pc0 = torch.tensor([0.0], dtype=torch.float64, device=self.device)
        self.y_pc1 = torch.tensor([1.0], dtype=torch.float64, device=self.device)

        #PDE parameters
        self.VF_mat1 = param['VF_mat1']
        self.VF_mat2 = param['VF_mat2']
        self.C_mat1  = param['C_mat1']
        self.C_mat2  = param['C_mat2']
        self.kappa = kappa

        ##NN
        self.layers = layers
        self.activation = nn.Tanh()
        self.best = np.inf

        #Networks initialization
        self.fcn_mat1 = FCN(self.layers, self.device, activation=self.activation)
        self.fcn_mat1.to(self.device)
        self.fcn_mat2 = FCN(self.layers, self.device, activation=self.activation)
        self.fcn_mat2.to(self.device)

        ### Parameters
        fcp_mat1_param = list(self.fcn_mat1.parameters())
        fcp_mat2_param = list(self.fcn_mat2.parameters())
        params = fcp_mat1_param + fcp_mat2_param

        self.loss_fn    = nn.MSELoss(reduction ='mean')
        self.optimizer  = SOAP(params, lr=0.001)
        self.scheduler  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.weights    = [torch.tensor(1.0, requires_grad=False), torch.tensor(1.0, requires_grad=False),
                           torch.tensor(1.0, requires_grad=False)]
        self.w_update_freq = 100
        self.scheduler_freq = 100
        
        #logs
        self.loss_log             = []
        self.loss_residual_log    = []
        self.loss_interface_log   = []
        self.loss_bc_log          = []
        self.w1 = []
        self.w2 = []
        self.w3 = []

        self.it = 0
        self.best_iter = 0
        self.log_every = 1
        self.pbar = None

    def forward_mat1(self, points):
        Chi_mat1_pred = self.fcn_mat1(points)
        return Chi_mat1_pred

    def forward_mat2(self, points):
        Chi_mat2_pred = self.fcn_mat2(points)
        return Chi_mat2_pred

    def Loss_bc(self):
        Chi_pred_mat1 = self.forward_mat1(self.y_pc0)
        Chi_pred_mat2 = self.forward_mat2(self.y_pc1)
        loss_bc = 0.5*((Chi_pred_mat1)**2 + (Chi_pred_mat2)**2)
        return loss_bc

    def Loss_residual(self):

        Chi_mat1_pred = self.forward_mat1(self.y_mat1)
        Chi_mat2_pred = self.forward_mat2(self.y_mat2)
        Chi_mat1_pred_y  = torch.autograd.grad(Chi_mat1_pred, self.y_mat1, grad_outputs=torch.ones_like(Chi_mat1_pred), create_graph=True)[0]
        Chi_mat1_pred_yy = torch.autograd.grad(Chi_mat1_pred_y, self.y_mat1, grad_outputs=torch.ones_like(Chi_mat1_pred_y), create_graph=True)[0]
        Chi_mat2_pred_y  = torch.autograd.grad(Chi_mat2_pred, self.y_mat2, grad_outputs=torch.ones_like(Chi_mat2_pred), create_graph=True)[0]
        Chi_mat2_pred_yy = torch.autograd.grad(Chi_mat2_pred_y, self.y_mat2, grad_outputs=torch.ones_like(Chi_mat2_pred_y), create_graph=True)[0]

        equ1 = self.C_mat1*Chi_mat1_pred_yy
        equ2 = self.C_mat2*Chi_mat2_pred_yy

        loss_collocation = self.loss_fn(equ1, self.null_mat1) + self.loss_fn(equ2, self.null_mat2)
        peso = 0.0
        loss_reg = peso*(self.loss_fn(Chi_mat1_pred, self.null_mat1) + self.loss_fn(Chi_mat2_pred, self.null_mat2))

        return loss_collocation + loss_reg

    def Loss_Interface_Imperfect(self):

        kappa_crit = 3.0*10**3
        Chi_mat1_pred = self.forward_mat1(self.y_inter)
        Chi_mat2_pred = self.forward_mat2(self.y_inter)
        Chi_mat1_pred_y = torch.autograd.grad(Chi_mat1_pred, self.y_inter, grad_outputs=torch.ones_like(Chi_mat1_pred),
                                              create_graph=True)[0]
        Chi_mat2_pred_y = torch.autograd.grad(Chi_mat2_pred, self.y_inter, grad_outputs=torch.ones_like(Chi_mat2_pred),
                                              create_graph=True)[0]

        jump = Chi_mat2_pred - Chi_mat1_pred
        eq1 = self.C_mat1*Chi_mat1_pred_y + self.C_mat1
        eq2 = self.C_mat2*Chi_mat2_pred_y + self.C_mat2

        if self.kappa < kappa_crit:
           equ1 = eq1 - (self.kappa * jump)
        else:
           equ1 = (eq1/self.kappa) - jump

        loss_interface1 = self.loss_fn(equ1, self.null_inter)
        loss_interface3 = self.loss_fn(eq1 - eq2, self.null_inter)
        loss_interface = loss_interface1 + loss_interface3 #+ loss_interface2
        return loss_interface

    def loss(self):
        loss_residual     = self.Loss_residual()
        loss_interface    = self.Loss_Interface_Imperfect( )
        loss_bc           = self.Loss_bc()
        return loss_residual, loss_interface, loss_bc

    def compute_gradients(self, loss_residual, loss_interface, loss_bc):
        params = list(self.fcn_mat1.parameters()) + list(self.fcn_mat2.parameters())
        grads_co = torch.autograd.grad(loss_residual, params, allow_unused=True)
        grads_it = torch.autograd.grad(loss_interface, params)
        grads_bc = torch.autograd.grad(loss_bc, params)
        return grads_co, grads_it, grads_bc

    def compute_weights(self):
        alpha = 0.9
        eps = 1e-18
        l_residual, l_interface, l_bc = self.loss()
        grads_co, grads_it, grads_bc = self.compute_gradients(l_residual, l_interface, l_bc)

        n_co = torch.linalg.norm(torch.cat([g.view(-1) for g in grads_co if g is not None]))
        n_it = torch.linalg.norm(torch.cat([g.view(-1) for g in grads_it]))
        n_bc = torch.linalg.norm(torch.cat([g.view(-1) for g in grads_bc]))
        del grads_co, grads_it, grads_bc

        total_norm = n_co + n_it + n_bc
        self.weights[0] = alpha * self.weights[0] + (1-alpha) * (total_norm / (n_co + eps))
        self.weights[1] = alpha * self.weights[1] + (1-alpha) * (total_norm / (n_it + eps))
        self.weights[2] = alpha * self.weights[2] + (1-alpha) * (total_norm / (n_bc + eps))

    def Log(self, loss_total, loss_residual, loss_interface, loss_bc, pbar, it, log_every):
        self.loss_log.append(loss_total.item())
        self.loss_residual_log.append(loss_residual.item())
        self.loss_interface_log.append(loss_interface.item())
        self.loss_bc_log.append(loss_bc.item())

        if it % log_every == 0:
           pbar.set_postfix({'Loss': loss_total.item(),
                            'Residual loss': loss_residual.item(),
                            'Interface loss': loss_interface.item(),
                            'BC loss': loss_bc.item()})

    def train(self, epochs = 30000, log_every=None, contrast = None):

        if log_every is not None:
           self.log_every = log_every
        else:
           self.log_every = 10

        self.pbar = trange(epochs)
        for it in self.pbar:

            self.it = it
            if it % self.w_update_freq == 0:
               self.compute_weights()
            w1, w2, w3 = self.weights[0].detach().item(), self.weights[1].detach().item(), self.weights[2].detach().item()
            self.w1.append(w1)
            self.w2.append(w2)
            self.w3.append(w3)

            self.optimizer.zero_grad()
            l_residual, l_interface, l_bc = self.loss()
            loss_total = w1*l_residual + w2*l_interface + w3*l_bc
            loss_total.backward()
            self.optimizer.step()

            if it % self.scheduler_freq == 0:
                self.scheduler.step()
            self.Log(loss_total, l_residual, l_interface, l_bc, self.pbar, it, self.log_every)

            if loss_total.item() < self.best:
                file_name = f"model:{date}_vf:{self.prob_name}_kappa:{self.kappa}_contrast:{contrast}.pth"
                file_path = os.path.join(save_dir, file_name)
                torch.save(self.state_dict(), file_path)
                self.best = loss_total.item()
                self.best_iter = it

        data = {'total loss': self.loss_log,
                'residual loss': self.loss_residual_log,
                'interface loss': self.loss_interface_log,
                'bc loss': self.loss_bc_log}

        weights = {'w_residual': self.w1,
                   'w_interface': self.w2,
                   'w_bc': self.w3}
         
        df = pd.DataFrame(weights)
        path_to_save_weights = os.path.join(save_dir, 
                               f'weights_{date}_{self.vf}_kappa:{self.kappa}_contrast:{contrast}.csv')
        df.to_csv(path_to_save_weights, index=False)

        return self.best_iter, data

    def Effective_Properties_Imperfect(self, points1, points2):
        y_mat1 = torch.as_tensor(points1, dtype=torch.float64, device=self.device).requires_grad_(True)
        y_mat2 = torch.as_tensor(points2, dtype=torch.float64, device=self.device).requires_grad_(True)

        Chi_mat1_pred = self.forward_mat1(y_mat1)
        Chi_mat2_pred = self.forward_mat2(y_mat2)

        Chi_mat1_pred_y = torch.autograd.grad(Chi_mat1_pred, y_mat1, grad_outputs=torch.ones_like(Chi_mat1_pred),
                                              create_graph=True)[0]
        Chi_mat2_pred_y = torch.autograd.grad(Chi_mat2_pred, y_mat2, grad_outputs=torch.ones_like(Chi_mat2_pred),
                                              create_graph=True)[0]

        Exp_mat1 = self.C_mat1 + self.C_mat1*Chi_mat1_pred_y
        Exp_mat2 = self.C_mat2 + self.C_mat2*Chi_mat2_pred_y
        integral1 = torch.trapezoid(Exp_mat1, y_mat1, dim=0)
        integral2 = torch.trapezoid(Exp_mat2, y_mat2, dim=0)
        C_eff = integral1 + integral2
        C_eff = C_eff.detach().cpu().numpy()

        denom = self.C_mat1 * self.C_mat2 + self.kappa * (self.C_mat2 * self.vf + self.C_mat1 * (1 - self.vf))
        C_eff_analytic = (self.C_mat1 * self.C_mat2 * self.kappa) / denom

        return C_eff[0], C_eff_analytic
    

def calcular_mn(C1, C2, kappa, yI):
    num1 = (C1 * C2) + kappa * (1 - yI) * (C1 - C2)
    den1 = (C1 * C2) + kappa * (C2 * yI + C1 * (1 - yI))
    m1 = -1 * (num1 / den1)

    num2 = -(C1 * C2) + kappa * yI * (C1 - C2)
    den2 = den1
    m2 = num2 / den2

    n1 = 0
    n2 = -m2
    return m1, m2, n1, n2

def calcular_chi(C1, C2, yI, y1, y2, kappa):
    m1, m2, n1, n2 = calcular_mn(C1, C2, kappa, yI)
    chi1 = m1 * y1 + n1
    chi2 = m2 * y2 + n2
    return chi1, chi2

#-----------------------------------------------------------------------------------------------------------
#---------------------Training loop for all cases with different contrasts and volume fractions---------------------
#-----------------------------------------------------------------------------------------------------------
EPOCHS = 30000
volum_frac = [0.2, 0.8]
volum_frac_str = [str(v) for v in volum_frac]
contrast_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
contrast_list_str = [str(c) for c in contrast_list]
kappa_list = [1, 10, 50, 100, 1000, 10000, 10**7]
best_iter_time_dic = {}
time_per_case = []
cmat2 = 12

for kappa in kappa_list:
    best_iter_time_dic[str(kappa)] = {}

    for contrast in contrast_list:
        C_mat1 = contrast*cmat2
        C_mat2 = cmat2
        best_iter_time_dic[str(kappa)][str(contrast)] = {}

        for vf, vf_str in zip(volum_frac, volum_frac_str):
            print(f"\n ---- For vf = {vf}, kappa = {kappa} and contrast = {contrast} ----")

            VF_mat1 = vf
            VF_mat2 = 1 - vf
            PARAM = {'VF_mat1':VF_mat1, 'VF_mat2':VF_mat2, 'C_mat1':C_mat1, 'C_mat2':C_mat2}
            y_interface = np.array([[vf]])

            #data for training
            N_POINTS = 500
            y_points1 = np.linspace(0, vf, N_POINTS )[:,None]
            y_points2 = np.linspace(vf, 1, N_POINTS )[:,None]
            y1_tensor = torch.as_tensor(y_points1, dtype=torch.float64, device=DEVICE).requires_grad_(True)
            y2_tensor = torch.as_tensor(y_points2, dtype=torch.float64, device=DEVICE).requires_grad_(True)

            model = PINN(y_collocation_mat1 = y1_tensor, y_collocation_mat2 = y2_tensor, 
                         y_interface = y_interface, param = PARAM, layers = LAYERS, 
                         kappa = kappa, device = DEVICE, prob_name = vf_str)
            model.to(DEVICE)

            t0 = time.perf_counter()
            bestIter, datos = model.train(epochs=EPOCHS, log_every=LOG_STEP, contrast = contrast)
            t1 = time.perf_counter()
            run_time = (t1 - t0)/60
            print(f"Tiempo de ejecucion por caso: {run_time:.3f} minutos")
            time_per_case.append(run_time)

            best_iter_time_dic[str(kappa)][str(contrast)][vf_str] = (bestIter, run_time)
            each_loss = {'Total loss': datos['total loss'],
                        'Residual loss': datos['residual loss'],
                        'Interface loss': datos['interface loss'],
                        'BC loss': datos['bc loss']}

            df = pd.DataFrame(each_loss)
            df.to_csv(os.path.join(save_dir, f'table_losses_{date}_contrast_{contrast}_vf_{vf_str}_kappa_{kappa}.csv'), index=False)

            fig, ax1 = plt.subplots(figsize=(8,6))
            df.plot(ax=ax1)
            ax1.set_xlabel('Epochs')
            ax1.set_yscale('log', base=10)
            ax1.set_ylabel('Losses')
            ax1.grid(True)
            ax1.legend(loc='upper right', fontsize=12)
            out_name = os.path.join(save_dir, f'all_losses_{date}_contrast_{contrast}_vf_{vf_str}_kappa_{kappa}.png')
            fig.savefig(out_name, dpi=300, bbox_inches='tight')
            plt.close(fig)

print(f"Tiempo promedio local: {sum(time_per_case)/len(time_per_case):.3f} minutos")
