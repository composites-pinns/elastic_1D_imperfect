import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def assemble_A_b(C1, C2, n1, n2, h1, h2, beta, nrep, uL=0.0, uR=1.0):
    '''
    Construye la matriz tridiagonal para la secuencia

        C1 --imp-- C2 --perf-- C1 --imp-- C2 --perf-- ... -- C1 --imp-- C2

    con nrep bloques materiales C1|C2.
    nrep   : número de repeticiones del par material C1|C2
    '''
    a1 = C1 / h1**2
    a2 = C2 / h2**2
    g1 = C1 / h1
    g2 = C2 / h2
    # -------------------------------------------------------
    # BLOQUE INICIAL: C1 | imp | C2
    # -------------------------------------------------------
    upper_first = np.concatenate([np.full(n1 - 1, -a1), np.array([-a1, -beta, -g2]), np.full(n2 - 1, -a2)])
    main_first = np.concatenate([np.full(n1, 2.0 * a1), np.array([g1 + beta, g2 + beta]), np.full(n2, 2.0 * a2)])
    lower_first = np.concatenate([np.full(n1 - 1, -a1), np.array([-g1, -beta, -a2]), np.full(n2 - 1, -a2)])
    # -------------------------------------------------------
    # BLOQUE REPETIDO: perf | C1 | imp | C2
    # Este bloque se agrega nrep-1 veces
    # -------------------------------------------------------
    upper_repeat = np.concatenate([np.array([-a2, -g1]), np.full(n1 - 1, -a1), np.array([-a1, -beta, -g2]), np.full(n2 - 1, -a2)])
    main_repeat = np.concatenate([np.array([g1 + g2]), np.full(n1, 2.0 * a1), np.array([g1 + beta, g2 + beta]), 
                                 np.full(n2, 2.0 * a2)])
    lower_repeat = np.concatenate([np.array([-g2, -a1]), np.full(n1 - 1, -a1), np.array([-g1, -beta, -a2]), np.full(n2 - 1, -a2)])

    main = np.concatenate([main_first, np.tile(main_repeat, nrep - 1)])
    upper = np.concatenate([upper_first, np.tile(upper_repeat, nrep - 1)])
    lower = np.concatenate([lower_first, np.tile(lower_repeat, nrep - 1)])
    A = np.diag(main) + np.diag(upper, 1) + np.diag(lower, -1)
    # -------------------------------------------------------
    # VECTOR b
    # -------------------------------------------------------
    N = main.size
    b = np.zeros(N, dtype=float)
    b[0] = a1 * uL
    b[-1] = a2 * uR
    return A, b

def unpack_solution(u, n1, n2, nrep):
    """
    Patrón global: C1 | imp(2) | C2 | perf(1) | C1 | imp(2) | C2 | ... | C2
    """
    out = {"C1": [], "imp": [], "C2": [], "perf": []}
    s = 0
    #primer bloque: C1 | imp | C2
    out["C1"].append(u[s:s+n1].copy()); s += n1
    out["imp"].append((u[s], u[s+1])); s += 2
    out["C2"].append(u[s:s+n2].copy()); s += n2

    #bloques restantes: perf | C1 | imp | C2
    for _ in range(nrep - 1):
        out["perf"].append(u[s]); s += 1
        out["C1"].append(u[s:s+n1].copy()); s += n1
        out["imp"].append((u[s], u[s+1])); s += 2
        out["C2"].append(u[s:s+n2].copy()); s += n2

    return out

def reconstruct_solution(u, n1, n2, h1, h2, nrep, uL, uR):
    blocks = unpack_solution(u, n1, n2, nrep)
    v1 = (n1 + 1) * h1
    v2 = (n2 + 1) * h2
    cell_length = v1 + v2

    points = [(0.0, uL, "boundary")]

    for cell in range(nrep):
        x_start = cell * cell_length

        x_c1 = x_start + h1 * np.arange(1, n1 + 1)
        u_c1 = blocks["C1"][cell]
        points.extend((x, val, "C1") for x, val in zip(x_c1, u_c1))

        x_imp = x_start + v1
        u_minus, u_plus = blocks["imp"][cell]
        points.append((x_imp, u_minus, "interface_left"))
        points.append((x_imp, u_plus, "interface_right"))

        x_c2 = x_imp + h2 * np.arange(1, n2 + 1)
        u_c2 = blocks["C2"][cell]
        points.extend((x, val, "C2") for x, val in zip(x_c2, u_c2))

        if cell < nrep - 1:
            x_perf = x_start + cell_length
            u_perf = blocks["perf"][cell]
            points.append((x_perf, u_perf, "interface_perfect"))

    points.append((nrep * cell_length, uR, "boundary"))
    return points

def build_plot_arrays_with_breaks(reconstruction):
    x = []
    y = []
    tags = []

    for xi, yi, tag in reconstruction:
        x.append(xi)
        y.append(yi)
        tags.append(tag)

        if tag == "interface_left":
            x.append(np.nan)
            y.append(np.nan)
            tags.append("break")

    return np.array(x, dtype=float), np.array(y, dtype=float), np.array(tags, dtype=object)

def q_exact(C1, C2, alpha, beta):
    return 1.0 / (alpha / C1 + (1.0 - alpha) / C2 + 1.0 / beta)

def build_exact_heterogeneous(C1, C2, alpha, beta, eps, nrep, n1=120, n2=120):
    q = q_exact(C1, C2, alpha, beta)

    x_all = [0.0]
    u_all = [0.0]

    for j in range(nrep):
        xj = j * eps
        xg = xj + alpha * eps
        xr = (j + 1) * eps

        base = q * j * eps * (alpha / C1 + (1.0 - alpha) / C2 + 1.0 / beta)

        x1 = np.linspace(xj, xg, n1, endpoint=False)
        u1 = base + q * (x1 - xj) / C1

        u_minus = base + q * alpha * eps / C1
        u_plus = u_minus + q * eps / beta

        x2 = np.linspace(xg, xr, n2 + 1, endpoint=True)[1:]
        u2 = base + q * alpha * eps / C1 + q * eps / beta + q * (x2 - xg) / C2

        x_all.extend(x1.tolist())
        u_all.extend(u1.tolist())
        x_all.append(xg)
        u_all.append(u_minus)
        x_all.append(np.nan)
        u_all.append(np.nan)
        x_all.append(xg)
        u_all.append(u_plus)
        x_all.extend(x2.tolist())
        u_all.extend(u2.tolist())

    x_all[-1] = 1.0
    u_all[-1] = 1.0

    return np.array(x_all, dtype=float), np.array(u_all, dtype=float)

def insert_break_at_interface(x_cell, u_cell, idx_interface):
    """
      Inserta NaN entre los dos lados de la interfaz imperfecta:
      x_cell[idx_interface]     = x_gamma, lado izquierdo
      x_cell[idx_interface + 1] = x_gamma, lado derecho
    """
    x_new = np.concatenate([x_cell[:idx_interface + 1],[np.nan], x_cell[idx_interface + 1:]])
    u_new = np.concatenate([u_cell[:idx_interface + 1],[np.nan], u_cell[idx_interface + 1:]])
    return x_new, u_new

if __name__ == "__main__":
    #Parámetros del material
    C2 = 12.0
    C1 = 5.0 * C2

    eps_list = [0.5, 0.25, 0.1, 0.01]
    alpha = 0.8

    #Discretización heterogénea/exacta
    n1_het = 120
    n2_het = 120

    #Discretización PINNs/local
    n_mid = 1000

    x0 = 0.0
    x1 = 1.0

    date = '2026_02_04'
    save_dir = r'C:\Users\boris\Desktop\yoanh\pinns\contrast\save'
    file_name = f'datos_contrast_{date}.npz'
    file_path_npz = os.path.join(save_dir, file_name)
    loaded = np.load(file_path_npz, allow_pickle=True)

    N_pinns = loaded['data1'].item()
    points_y = loaded['data2'].item()
    N_analy_dict = loaded['data4'].item()

    kappa_list = [1, 10, 50, 100, 1000, 10**7]
    kappa_leyenda = ['1', '10', '50', r'10^2', r'10^3', r'10^7']
    vf_str = str(alpha)
    target_contrast = "5.0"

    #lista de colores para los distintos kappas
    color_map = plt.get_cmap('tab10')

    #lista para los errores
    error_results = []

    for eps in eps_list:
        v1 = alpha * eps
        v2 = (1 - alpha) * eps
        h1 = v1 / (n1_het + 1)
        h2 = v2 / (n2_het + 1)
        nrep = int(round(1 / eps))

        plt.figure(figsize=(12, 8))
        
        for k_idx, kappa in enumerate(kappa_list):
            print(f"Procesando eps={eps}, kappa={kappa}...")
            
            # Datos del problema local
            y_pinn = points_y[str(kappa)][target_contrast][vf_str]
            N_pinn = N_pinns[str(kappa)][target_contrast][vf_str]
            N_analy = N_analy_dict[str(kappa)][target_contrast][vf_str].flatten()
    
            # índice de la interfaz en PINNs
            idx_interface_pinn = n_mid - 1
    
            #Reconstrucción con PINNs y analítica
            x_plot_pinn = []
            u_plot_pinn = []
            u_plot_analy = []
    
            for cell in range(nrep):
                x_start = cell * eps
                x_cell = x_start + eps * y_pinn
                u_cell_pinn = x_cell + eps * N_pinn
                u_cell_analy = x_cell + eps * N_analy
    
                x_cell_b, u_cell_pinn_b = insert_break_at_interface(x_cell, u_cell_pinn, idx_interface_pinn)
                _, u_cell_analy_b = insert_break_at_interface(x_cell, u_cell_analy, idx_interface_pinn)
    
                x_plot_pinn.extend(x_cell_b.tolist())
                u_plot_pinn.extend(u_cell_pinn_b.tolist())
                u_plot_analy.extend(u_cell_analy_b.tolist())
    
            x_plot_pinn = np.array(x_plot_pinn, dtype=float)
            u_plot_pinn = np.array(u_plot_pinn, dtype=float)
            u_plot_analy = np.array(u_plot_analy, dtype=float)

            #errores
            mask = ~np.isnan(u_plot_pinn) #& ~np.isnan(u_plot_analy)
            abs_diff = np.abs(u_plot_pinn[mask] - u_plot_analy[mask])
            mae      = np.mean(abs_diff)
            rmse     = np.sqrt(np.mean(abs_diff**2))
            max_err  = np.max(abs_diff)
            
            #Guardar resultados
            error_results.append({'eps': eps,'kappa': kappa,'mae': mae,'rmse': rmse,
                                  'max_error': max_err})
            
            current_color = color_map(k_idx)
            plt.plot(x_plot_pinn, u_plot_pinn, '-', color=current_color, 
                     label=rf'$u_{{\text{{AHM-PINNs}}}}^{{(\varepsilon)}}$ ($\kappa={kappa_leyenda[k_idx]}$)', linewidth=1.0)
            plt.plot(x_plot_pinn, u_plot_analy, '--', color=current_color, 
                     label=rf'$u_{{\text{{AHM-analytic}}}}^{{(\varepsilon)}}$ ($\kappa={kappa_leyenda[k_idx]}$)', linewidth=2.0)

        plt.xlabel('x', fontsize=12)
        plt.ylabel(r'$u^{(\epsilon)}(x)$', fontsize=12)
        plt.title(rf'$\epsilon$={eps}', fontsize=14)
        plt.xticks(np.arange(0.0, 1.1, 0.1))
        plt.xlim(0, 1.0)
        plt.grid(True, alpha=0.3)
        
        # Ajustar leyenda
        plt.legend(bbox_to_anchor=(0.25, 1), loc='upper right', fontsize=11, frameon=False)
        plt.tight_layout()
        
        # Guardar
        save_solution_dir = r'C:\Users\boris\Desktop\yoanh\pinns\contrast\figuras_reconstruccion'
        if not os.path.exists(save_solution_dir):
            os.makedirs(save_solution_dir)
            
        path_to_save = os.path.join(save_solution_dir, f'unida_eps_{eps}_todas_kappas_vf_{alpha}.png')
        path_to_save = os.path.join(save_solution_dir, f'unida_eps_{eps}_todas_kappas_vf_{alpha}.pdf')
        plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
        plt.savefig(path_to_save, format='pdf', bbox_inches='tight')
        
        print(f"Figura guardada en: {path_to_save}")
        plt.close()

    # Guardar la tabla de errores
    df_errors = pd.DataFrame(error_results)
    save_error_dir = r'C:\Users\boris\Desktop\yoanh\pinns\contrast\errores_reconstruccion'
    if not os.path.exists(save_error_dir):
        os.makedirs(save_error_dir)
        
    save_error_path = os.path.join(save_error_dir, 'errores_reconstruccion.csv')
    df_errors.to_csv(save_error_path, index=False)
    print(f"\nTabla de errores guardada en: {save_error_path}")
    print(df_errors)