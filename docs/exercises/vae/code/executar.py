"""Executa na ordem os exercícios, sem reutilizar dados de teste para decisões de treino.
Uso: python code/executar.py (a partir de docs/exercises/vae/).
"""
import json
import random
import numpy as np
import torch
from pathlib import Path
from ex1a import carregar_dados
from ex1c import train_model, run_epoch, fig1, DEVICE
from ex1d import fig2, fig3
from ex2a import treinar_vae_2d
from ex2b import fig4
from ex2c import fig5
from ex2d import fig6
from relatorio import replace_report

ROOT = Path(__file__).resolve().parent.parent

def main():
    print('Dispositivo:', DEVICE, flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    train_loader, val_loader, test_loader, test = carregar_dados()
    model16, hist16 = train_model(16, train_loader, val_loader)
    rec16, kl16 = run_epoch(model16, test_loader)
    fig1(hist16)
    selected = fig2(model16, test)
    fig3(model16)
    model2, hist2, rec2, kl2 = treinar_vae_2d(train_loader, val_loader, test_loader)
    fig4(model2, test_loader)
    fig5(model2)
    indices, digits = fig6(model16, test)
    metrics = {'d16': {'rec': rec16, 'kl': kl16, 'total': rec16 + kl16},
               'd2': {'rec': rec2, 'kl': kl2, 'total': rec2 + kl2},
               'val_kl_first': hist16['val_kl'][0], 'val_kl_last': hist16['val_kl'][-1],
               'indices': indices, 'digits': digits, 'first_digit_indices': selected,
               'history16': hist16, 'history2': hist2}
    (ROOT / 'resultados.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    replace_report(metrics)
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith('history')}, indent=2))
    print('Figuras e relatório atualizados.', flush=True)

if __name__ == '__main__':
    main()
