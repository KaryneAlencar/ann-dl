import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ex1c import DEVICE, FIG
from ex1d import display_img

def fig6(model, test):
    # Seleciona duas imagens de teste de classes diferentes: 7 e 1
    ia, ib = 0, 2
    xa, ya = test[ia]
    xb, yb = test[ib]
    assert int(ya) != int(yb)
    xa, xb = xa.to(DEVICE).flatten(), xb.to(DEVICE).flatten()
    
    # Define dez passos igualmente espaçados entre t=0 e t=1
    ts = torch.linspace(0, 1, 10, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        # Obtém as médias latentes das duas imagens
        mua, _ = model.encode(xa.unsqueeze(0))
        mub, _ = model.encode(xb.unsqueeze(0))

        # Interpola diretamente as intensidades dos pixels
        pixel_blend = ((1-ts)*xa + ts*xb).cpu().numpy()
        
        # Interpola entre as médias latentes e decodifica os resultados
        latent_blend = model.decoder((1-ts)*mua + ts*mub).cpu().numpy()
    
    # Compara as duas interpolações em linhas de dez imagens
    fig, axs = plt.subplots(2, 10, figsize=(14, 3.5))
    for j in range(10):
        display_img(axs[0, j], pixel_blend[j])
        display_img(axs[1, j], latent_blend[j])
        axs[0, j].set_title(f't={j/9:.2f}', fontsize=9)
    axs[0, 0].set_ylabel('Pixels')
    axs[1, 0].set_ylabel('Latente')
    fig.suptitle(f'Figura 6 — Interpolacao dos indices {ia} (digito {int(ya)}) e {ib} (digito {int(yb)})')
    fig.tight_layout()
    fig.savefig(FIG/'fig6.png', dpi=160)
    plt.close(fig)
    return [ia, ib], [int(ya), int(yb)]
