"""Insere os números obtidos no relatório (sem inventar resultados)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def replace_report(metrics):
    report = ROOT / 'index.md'
    content = report.read_text(encoding='utf-8')
    replacements = {
        'PENDENTE_REC16': f"{metrics['d16']['rec']:.3f}",
        'PENDENTE_KL16': f"{metrics['d16']['kl']:.3f}",
        'PENDENTE_TOTAL16': f"{metrics['d16']['total']:.3f}",
        'PENDENTE_VAL_INICIAL': f"{metrics['val_kl_first']:.3f}",
        'PENDENTE_VAL_FINAL': f"{metrics['val_kl_last']:.3f}",
        'PENDENTE_REC2': f"{metrics['d2']['rec']:.3f}",
        'PENDENTE_KL2': f"{metrics['d2']['kl']:.3f}",
        'PENDENTE_INDICES': f"{metrics['indices'][0]} e {metrics['indices'][1]}",
    }
    for k, v in replacements.items():
        content = content.replace(k, v)
    report.write_text(content, encoding='utf-8')
