import os
from os import path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from collections import defaultdict

plt.rcParams['mathtext.fontset'] = 'cm'   # Computer Modern
plt.rcParams['font.family'] = 'serif'

csv_path = r'C:\Users\boris\Desktop\yoanh\pinns\contrast\save_xxx'
figure_path = r'C:\Users\boris\Desktop\yoanh\pinns\contrast\figures_datos_xxx'
df = pd.read_csv(path.join(csv_path, "table_contrast_2026_02_04.csv"))

results = defaultdict(lambda: defaultdict(dict))

for _, row in df.iterrows():
    kappa = str(row["kappa"])
    contrast = str(row["contrast"])
    vf = str(row["vf"])

    results[kappa][contrast][vf] = {"C_eff_pred": row["C_eff_pred"],
                                    "C_eff_exact": row["C_eff_exact"]}

vf_values = [0.2, 0.5, 0.8]
vf_list = [str(vf) for vf in vf_values]
kappas = [1, 10, 50, 100, 1000,  10**7]
kappa_list = [str(kappa) for kappa in kappas]
kappas_to_print = ["1", "10", "50", "10^2", "10^3", "10^7"]

contrast_values = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
contrast_list = [str(c) for c in contrast_values]

for j, vf in enumerate(vf_list):

    fig, ax = plt.subplots(figsize=(8,6))

    all_y_vf = []
    y_offsets = [ [-0.4, -0.4, -0.4, -0.4, -0.4, 0.4],
                  [-0.6, -0.6, -0.6, -0.6, -0.6, 0.6],
                  [-1, -1, -1, -1, -1, 1] ]

    x_offsets = [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2]

    cmap_blue = plt.get_cmap('Blues')
    cmap_reds = plt.get_cmap('Reds')
    
    for i, (kappa, kappa_print) in enumerate(zip(kappa_list, kappas_to_print)):

        y_pred = [results[kappa][contrast][vf]["C_eff_pred"] for contrast in contrast_list]
        y_exact = [results[kappa][contrast][vf]["C_eff_exact"] for contrast in contrast_list]

        all_y_vf.extend(y_pred)
        all_y_vf.extend(y_exact)
        
        color_val = 0.4 + 0.6 * (i /(len(kappa_list) - 1))
        blue_color = cmap_blue(color_val)
        red_color  = cmap_reds(color_val)

        ax.plot(contrast_values, y_pred, linestyle='-', marker='o', color=blue_color, markersize=4, linewidth=3.0)
        ax.plot(contrast_values, y_exact, linestyle='--', color=red_color, linewidth=2.0, dashes=(2, 2))
        
        y_text = y_pred[-1] + y_offsets[j][i]
        
        ax.text(contrast_values[-1] + x_offsets[i],
                y_text, rf"$\kappa={kappa_print}$",
                fontsize=9, ha='left', va='center')

    rep_blue = cmap_blue(0.7)
    rep_red  = cmap_reds(0.7)
    
    legend_elements = [ Line2D([0], [0], color=rep_blue, lw=2.5, linestyle='-', marker='o', markersize=4, label=r'PINNs (continuous)'),
                        Line2D([0], [0], color=rep_red, lw=2.0, linestyle='--', dashes=(2, 2), label=r'Analytical (dashed)')]
    ax.legend(handles=legend_elements, loc='upper left', ncol=2)

    ax.set_title(rf"$V_1={vf}$")
    ax.set_xlabel(r"$E_1/E_2$")
    y_min = min(all_y_vf)
    y_max = max(all_y_vf)
    margin = 0.15 * (y_max - y_min) if y_max > y_min else 1e-6
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_ylabel("Effective Young's moduli")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outname_eff = path.join(figure_path, f'effective_properties_vf_{vf}.png')
    outname_eff1 = path.join(figure_path, f'effective_properties_vf_{vf}.pdf')
    plt.savefig(outname_eff, bbox_inches='tight')
    plt.savefig(outname_eff1, bbox_inches='tight')
    plt.show()

npz_file = path.join(csv_path, "datos_contrast_2026_02_04.npz")
loaded = np.load(npz_file, allow_pickle=True)
N_pinns = loaded['data1'].item()
points_y = loaded['data2'].item()
N_analy = loaded['data4'].item()


target_kappas = ["1", "100", "10000000"]
titles_kappas = [r"$\kappa=1$", r"$\kappa=10^2$", r"$\kappa=10^7$"]
target_contrast = "5.0"
vfs = [0.2, 0.5, 0.8] 

colors = plt.cm.plasma(np.linspace(0, 0.8, 6))

for idx, (kappa, title) in enumerate(zip(target_kappas, titles_kappas)):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    i = 0
    for vf in vfs:
        vf_str = str(vf)
        
        y_pts = points_y[kappa][target_contrast][vf_str].flatten()
        y_an = N_analy[kappa][target_contrast][vf_str].flatten()
        y_pn = N_pinns[kappa][target_contrast][vf_str].flatten()
        error = np.abs(y_an - y_pn)
        
        axes[0].plot(y_pts, y_pn, linestyle='-', linewidth=2.5, color=colors[i])
        axes[0].plot(y_pts, y_an, linestyle='--', linewidth=2.5, color=colors[i+1], dashes=(3, 2))

        axes[1].plot(y_pts, error, linewidth=2, color=colors[i], label=rf'$V_1={vf}$')
        i += 2

    axes[0].text(0.04, 0.04, title, transform=axes[0].transAxes, fontsize=12, 
                 verticalalignment='bottom')
                 
    axes[0].set_xlabel(r'$y\quad (displacement)$')
    axes[0].set_ylabel('Local problem solutions')
    axes[0].grid(True, alpha=0.3)

    #axes[1].set_title("Absolute Error")
    axes[1].set_xlabel(r'$y\quad (displacement)$')
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[1].set_ylabel(r'Absolute Error ($e$)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   frameon=False, borderaxespad=0, ncol=1)

    axes[0].legend([Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5),
                    Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5)],
                   [r'PINNs (continuous)', r'Analytical (dashed)'],
                   loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
                   
    outname = path.join(figure_path, f'pseudo_disp_analytical_vs_pinns_kappa_{kappa}_{target_contrast}.png')
    outname_pdf = path.join(figure_path, f'pseudo_disp_analytical_vs_pinns_kappa_{kappa}_{target_contrast}.pdf')
    plt.savefig(outname, bbox_inches='tight')
    plt.savefig(outname_pdf, bbox_inches='tight')

plt.tight_layout()
plt.show()
