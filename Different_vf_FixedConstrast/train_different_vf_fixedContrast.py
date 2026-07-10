import sys
import os
from tqdm import trange
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimizer.soap import SOAP

print(torch.get_default_dtype())
torch.set_default_dtype(torch.float64)
print(torch.get_default_dtype())

path_soap = '/home/bmederos/pinns'
sys.path.append(path_soap)
save_dir = '/home/bmederos/pinns/save_soap'

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
EPOCHS = 30000  # 1500 # 1000

LOG_STEP = 5
date = '2026_02_04'
train_flag = False

y_I = 0.2 # 0.3,0.2

VF_mat1 = y_I
VF_mat2 = 1 - VF_mat1

C_mat1 = 45 # 45, 3.4
C_mat2 = 12 # 12, 85

PARAM = {'VF_mat1':VF_mat1, 'VF_mat2':VF_mat2, 'C_mat1':C_mat1, 'C_mat2':C_mat2}

N_COLLOCATION = 500 #100
N_UNIQUENESS  = 1000
N_BOUNDARY = 1

# Interface
y_interface = np.array([[y_I]])

# Uniqueness
y_uniqueness1 = np.linspace(0, y_I, N_UNIQUENESS)[:,None]
y_uniqueness2 = np.linspace(y_I, 1, N_UNIQUENESS)[:,None]

N_POINTS = 1000

y_points1 = np.linspace(0,    y_I, N_POINTS )[:,None]
y_points2 = np.linspace(y_I,  1,   N_POINTS )[:,None]

y_points_exact = np.concatenate((y_points1,y_points2))

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
    
class PINNSampler:
    def __init__(self, a, b, n_points, device):
        self.a = a
        self.b = b
        self.n_points = n_points
        self.device = device

    def sample(self):
        #muestreo uniforme en [a,b]
        x = self.a + torch.rand(self.n_points, 1, device=self.device, requires_grad=True) * (self.b - self.a)
        return 
    
class PINN(nn.Module):

    def __init__(self, y_collocation_mat1 = None, y_collocation_mat2 = None, y_interface = None, param = None,
                 layers = None, kappa = None, device = None, prob_name = None):

        super().__init__()

        self.prob_name = prob_name
        if y_interface != None:
           self.vf = y_interface[0,0]
        self.device = device

        if y_collocation_mat1 != None:
           self.y_mat1 = torch.tensor(y_collocation_mat1, requires_grad= True).double().to(self.device)
           self.null_mat1 = torch.zeros(self.y_mat1.shape, dtype = torch.float64, requires_grad = False).to(self.device)

        if y_collocation_mat2 != None:
           self.y_mat2 = torch.tensor(y_collocation_mat2, requires_grad= True).double().to(self.device)
           self.null_mat2 = torch.zeros(self.y_mat2.shape, dtype = torch.float64, requires_grad = False).to(self.device)

        # Interface points
        if y_interface != None:
           self.y_inter = torch.tensor(y_interface, requires_grad= True).double().to(self.device)
           self.null_inter = torch.zeros(self.y_inter.shape, dtype = torch.float64, requires_grad=False).to(self.device)

        #boundary points
        self.y_pc0 = torch.tensor([0.0], requires_grad=False).to(self.device)
        self.y_pc1 = torch.tensor([1.0], requires_grad=False).to(self.device)

        #Interface points
        self.y_inter = torch.tensor(y_interface, requires_grad= True).to(self.device)
        self.null_inter = torch.zeros(self.y_inter.shape, requires_grad=False).to(self.device)

        #PDE parameters
        self.VF_mat1 = param['VF_mat1']
        self.VF_mat2 = param['VF_mat2']
        self.C_mat1  = param['C_mat1']
        self.C_mat2  = param['C_mat2']
        self.kappa = kappa

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
        self.fcn_mat1 = FCN(self.layers,self.device, activation=self.activation)
        self.fcn_mat1.to(self.device)
        self.fcn_mat2 = FCN(self.layers,self.device, activation=self.activation)
        self.fcn_mat2.to(self.device)

        ### Parameters
        # Get the parameters as separate lists
        fcp_mat1_param = list(self.fcn_mat1.parameters())
        fcp_mat2_param = list(self.fcn_mat2.parameters())
        params = fcp_mat1_param + fcp_mat2_param

        self.optimizer  = SOAP(params, lr=0.001)
        self.scheduler  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.weights = [torch.tensor(1.0, requires_grad=False), torch.tensor(1.0, requires_grad=False),
                        torch.tensor(1.0, requires_grad=False)]
        self.w_update_freq = 100

        self.loss_fn = nn.MSELoss(reduction ='mean')
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
        #Predicted
        Chi_pred_mat1 = self.forward_mat1(self.y_pc0)
        Chi_pred_mat2 = self.forward_mat2(self.y_pc1)
        # Compute losses
        loss_bc = 0.5*((Chi_pred_mat1)**2 + (Chi_pred_mat2)**2)
        return loss_bc

    def Loss_residual(self):

        #Predicted
        Chi_mat1_pred = self.forward_mat1(self.y_mat1)
        Chi_mat2_pred = self.forward_mat2(self.y_mat2)

        Chi_mat1_pred_y  = torch.autograd.grad(Chi_mat1_pred, self.y_mat1, grad_outputs=torch.ones_like(Chi_mat1_pred), create_graph=True)[0]
        Chi_mat1_pred_yy = torch.autograd.grad(Chi_mat1_pred_y, self.y_mat1, grad_outputs=torch.ones_like(Chi_mat1_pred_y), create_graph=True)[0]

        Chi_mat2_pred_y  = torch.autograd.grad( Chi_mat2_pred, self.y_mat2, grad_outputs=torch.ones_like(Chi_mat2_pred), create_graph=True)[0]
        Chi_mat2_pred_yy = torch.autograd.grad(Chi_mat2_pred_y, self.y_mat2, grad_outputs=torch.ones_like(Chi_mat2_pred_y), create_graph=True)[0]

        equ1 = self.C_mat1*Chi_mat1_pred_yy
        equ2 = self.C_mat2*Chi_mat2_pred_yy

        self.null_mat1 = torch.zeros(self.y_mat1.shape, requires_grad = False).to(self.device)
        self.null_mat2 = torch.zeros(self.y_mat2.shape, requires_grad = False).to(self.device)

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
        #collocations
        grads_co = torch.autograd.grad(loss_residual, params, allow_unused=True)
        #interface
        grads_it = torch.autograd.grad(loss_interface, params)
        #uniqueness
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
        # Store losses
        self.loss_log.append(loss_total.item())
        self.loss_residual_log.append(loss_residual.item())
        self.loss_interface_log.append(loss_interface.item())
        self.loss_bc_log.append(loss_bc.item())

        # Log information
        if it % log_every == 0:
           pbar.set_postfix({'Loss': loss_total.item(),
                            'Residual loss': loss_residual.item(),
                            'Interface loss': loss_interface.item(),
                            'BC loss': loss_bc.item()})

    def train(self, epochs = 30000, log_every=10):

        self.log_every = log_every
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

            if it % 100 == 0:
                self.scheduler.step()
            self.Log(loss_total, l_residual, l_interface, l_bc, self.pbar, it, self.log_every)

            if loss_total.item() < self.best:

                file_name = f"model_laminated1d_pinns_date:{date}_vf:{self.prob_name}_kappa:{self.kappa}.pth"
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
         
        df = pd.DataFrame(data)
        path_to_save = os.path.join(save_dir, f'losses_{date}_{self.vf}_kappa:{self.kappa}.csv')
        df.to_csv(path_to_save, index=False)

        df1 = pd.DataFrame(weights)
        path_to_save_weights = os.path.join(save_dir, f'weights_{date}_{self.vf}_kappa:{self.kappa}.csv')
        df1.to_csv(path_to_save_weights, index=False)

        return self.best_iter, data

    def Effective_Properties_Imperfect(self, points1, points2):
        y_mat1 = torch.tensor(points1, requires_grad=True).to(DEVICE)
        y_mat2 = torch.tensor(points2, requires_grad=True).to(DEVICE)

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


import time
train_flag = True
EPOCHS = 30000

if train_flag:

    volum_frac = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    volum_frac_str = [str(v) for v in volum_frac]

    kappa_list = [1, 10, 50, 100, 1000, 10000, 10**7]
    best_iter_dic = {}

    for kappa in kappa_list:

        best_iter_dic[str(kappa)] = {}

        tt0 = time.perf_counter()
        for vf, vf_str in zip(volum_frac, volum_frac_str):
            print(f"\n ---- For vf = {vf} and kappa = {kappa} ----")

            VF_mat1 = vf
            VF_mat2 = 1 - vf
            C_mat1 = 45
            C_mat2 = 12
            PARAM = {'VF_mat1':VF_mat1, 'VF_mat2':VF_mat2, 'C_mat1':C_mat1, 'C_mat2':C_mat2}
            y_interface = np.array([[vf]])

            #data for training
            N_POINTS = 500
            y_points1 = np.linspace(0, vf, N_POINTS )[:,None]
            y_points2 = np.linspace(vf, 1, N_POINTS )[:,None]
            # Changed .float() to .double() to match default dtype for model parameters
            y1_tensor = torch.tensor(y_points1, requires_grad=True).double().to(DEVICE)
            y2_tensor = torch.tensor(y_points2, requires_grad=True).double().to(DEVICE)

            model = PINN(y_collocation_mat1 = y1_tensor, y_collocation_mat2 = y2_tensor, y_interface = y_interface,
                          param = PARAM, layers = LAYERS, kappa = kappa, device = DEVICE, prob_name = vf_str)
            model.to(DEVICE)

            t0 = time.perf_counter()
            bestIter, datos = model.train(epochs=EPOCHS, log_every=LOG_STEP)
            t1 = time.perf_counter()
            print(f"Tiempo de ejecucion por fraccion de volumen: {(t1 - t0)/60:.3f} minutos")

            best_iter_dic[str(kappa)][vf_str] = bestIter

            file_name  = f"model_laminated1d_pinns_date:{date}_vf:{vf_str}_kappa:{kappa}.pth" # Use current_kappa in filename
            file_path  = os.path.join(save_dir, file_name)
            torch.save(model.state_dict(), file_path) # Added missing save
            model.load_state_dict(torch.load(file_path))
            model.to(DEVICE)
            model.eval()

            each_loss = {'Total loss': datos['total loss'],
                          'Residual loss': datos['residual loss'],
                          'Interface loss': datos['interface loss'],
                          'BC loss': datos['bc loss']}

            df = pd.DataFrame(each_loss)
            df.to_csv(os.path.join(save_dir, f'table_losses_total_col_bc_interf{date}_{vf_str}_{kappa}.csv'), index=False)

            '''
            ax1 = df.plot(figsize=(8,6))
            #plt.ylim(0, 1)
            plt.xlabel('Epochs')
            plt.yscale('log', base=10)
            plt.ylabel('Losses')
            out_name = os.path.join(save_dir, f'all_losses_{date}_{vf_str}_{current_kappa}.png')
            plt.savefig(out_name)
            plt.grid(True)
            ax1.legend(loc='upper right', fontsize=12)
            plt.show()
            '''

    tt1 = time.perf_counter()
    print(f"Tiempo de ejecucion total: {(tt1 - tt0)/60:.3f} minutos")

    with open(os.path.join(save_dir, "best_iters{date}_{vf_str}_{kappa}.pkl"), "wb") as f:
        pickle.dump(best_iter_dic, f)


date = '2026_02_04'
volum_frac = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
volum_frac_str = [str(v) for v in volum_frac]
kappa_list = [1, 10, 50, 100, 1000, 10000, 10**7]

N_pinns    = {}
N_analy    = {}
points_y   = {}
table_rows = {}

for kappa in kappa_list:

    models = [f'model_laminated1d_pinns_date:{date}_vf:0.1_kappa:{kappa}.pth',
              f'model_laminated1d_pinns_date:{date}_vf:0.2_kappa:{kappa}.pth',
              f'model_laminated1d_pinns_date:{date}_vf:0.3_kappa:{kappa}.pth',
              f'model_laminated1d_pinns_date:{date}_vf:0.4_kappa:{kappa}.pth',
              f'model_laminated1d_pinns_date:{date}_vf:0.5_kappa:{kappa}.pth',
              f'model_laminated1d_pinns_date:{date}_vf:0.6_kappa:{kappa}.pth',
              f'model_laminated1d_pinns_date:{date}_vf:0.7_kappa:{kappa}.pth',
              f'model_laminated1d_pinns_date:{date}_vf:0.8_kappa:{kappa}.pth',
              f'model_laminated1d_pinns_date:{date}_vf:0.9_kappa:{kappa}.pth']
    

    N_pinns[str(kappa)]    = {}
    N_analy[str(kappa)]    = {}
    points_y[str(kappa)]   = {}
    table_rows[str(kappa)] = []

    for vf, vf_str, model_name in zip(volum_frac, volum_frac_str, models):

        VF_mat1 = vf
        VF_mat2 = 1 - vf
        C_mat1 = 45
        C_mat2 = 12
        PARAM = {'VF_mat1':VF_mat1, 'VF_mat2':VF_mat2, 'C_mat1':C_mat1, 'C_mat2':C_mat2}

        y_interface = np.array([[vf]])
        file_path = os.path.join(save_dir, model_name)
        print(file_path)

        model = PINN(y_interface = y_interface, param = PARAM, layers = LAYERS, kappa = kappa, device = DEVICE, prob_name = vf)
        model.load_state_dict(torch.load(file_path))
        model.to(DEVICE)

        #data for predictions of effective properties
        N_POINTS = 1000
        y_points1 = np.linspace(0, vf, N_POINTS )[:,None]
        y_points2 = np.linspace(vf, 1, N_POINTS )[:,None]
        y1 = y_points1
        y2 = y_points2
        #predict
        y1_tensor = torch.tensor(y1, requires_grad=True).to(DEVICE)
        y2_tensor = torch.tensor(y2, requires_grad=True).to(DEVICE)

        #propiedades efectivas
        C_eff_pred, C_eff_exact = model.Effective_Properties_Imperfect(y_points1, y_points2)

        #data for predictions of local displacements
        N_POINTS = 500
        y_points1 = np.linspace(0, vf, N_POINTS )[:,None]
        y_points2 = np.linspace(vf, 1, N_POINTS )[:,None]
        y1 = y_points1
        y2 = y_points2
        #predict
        y1_tensor = torch.tensor(y1, requires_grad=True).to(DEVICE)
        y2_tensor = torch.tensor(y2, requires_grad=True).to(DEVICE)

        #estimated
        N_mat1 = model.forward_mat1(y1_tensor)
        N_mat1 = N_mat1.detach().cpu().numpy()
        N_mat2 = model.forward_mat2(y2_tensor)
        N_mat2 = N_mat2.detach().cpu().numpy()
        y = np.concatenate([y1.flatten(), y2.flatten()])
        N_mat = np.concatenate([N_mat1.flatten(), N_mat2.flatten()])
        N_pinns[str(kappa)][vf_str] = N_mat
        points_y[str(kappa)][vf_str] = y

        #analytic
        n_points = 500
        c1, c2 = C_mat1, C_mat2
        y1 = np.linspace(0, vf, n_points)
        y2 = np.linspace(vf, 1, n_points)
        N1, N2 = calcular_chi(c1, c2, vf, kappa, y1, y2)
        N = np.concatenate([N1, N2])
        y = np.concatenate([y1, y2])
        N_analy[str(kappa)][vf_str] = N

        table_rows[str(kappa)].append({ 'vf': vf,
                                        'C_eff_pred':  C_eff_pred,
                                        'C_eff_exact': C_eff_exact,
                                        'relative_error': (abs(C_eff_pred - C_eff_exact)/C_eff_exact)*100})
