# Resultados del análisis

Métricas y figuras de la evaluación del pipeline (inpainting LaMa + realce A-ESRGAN).

## 1. LaMa — evaluación sobre dataset sintético (split test, 70 tríos)

Métrica de referencia completa sobre la zona enmascarada, preentrenado vs fine-tuned:

| Métrica | Preentrenado | Fine-tuned |
|---|---|---|
| PSNR (zona enmascarada) | ~20.57 dB | **~21.13 dB** |
| PSNR (imagen completa) | 24.23 dB | 24.38 dB |
| SSIM (imagen completa) | 0.661 | 0.666 |

El fine-tuning con máscara verdad-de-terreno produce una mejora **consistente pero
modesta** en la región reparada.

## 2. Pipeline completo — evaluación ciega sobre fotografía histórica real

Conjunto: 21 fotografías históricas (filtradas de 54 candidatas; se descartan 33 con
degradación de tipo moderno). Sin *ground truth* → métricas **no-referenciadas**.
Cobertura media de máscara: 5.5 %. Comparación *Original degradada* → *Pipeline
preentrenado* → *Pipeline ajustado* (media):

| Métrica | Dirección | Original | Preentr. | Ajustado | Wilcoxon (orig vs ajustado) |
|---|:--:|--:|--:|--:|---|
| NIQE    | ↓ | 4.01 | 4.14 | 4.25 | n.s. (p_Bonf 0.77) |
| BRISQUE | ↓ | 19.11 | 17.89 | 15.71 | n.s. (p_Bonf 1.0) |
| Ma      | ↑ | 7.29 | 5.90 | 5.73 | **significativo** (p_Bonf 7e-4) |
| PI      | ↓ | 3.36 | 4.12 | 4.26 | **significativo** (p_Bonf 0.019) |
| MUSIQ   | ↑ | 49.73 | 50.47 | 49.07 | n.s. (p_Bonf 1.0) |

**Lectura:** sobre fotografía histórica real, el pipeline **no** muestra una mejora
clara en calidad percibida no-referenciada. BRISQUE baja (mejor) de forma no
significativa; NIQE, Ma y PI empeoran, y en Ma/PI el empeoramiento es
estadísticamente significativo. Coherente con que A-ESRGAN se entrenó sobre un
dominio HR (DIV2K toneado) distinto al de estas fotos, y con la baja cobertura de
máscara (el inpainting apenas interviene). Detalle por imagen en los CSV.

## Archivos (poblar desde `Fase6/` de Drive — ver `scripts/publicar_datos.sh`)

```
fase6_evaluacion_historica/
  resultados_6.json                     resumen agregado + tests de Wilcoxon
  conjunto_6.csv                        composición del conjunto (tamaños, cobertura)
  metricas_ciegas_por_imagen_6.csv      NIQE/BRISQUE/Ma/MUSIQ/PI por imagen y condición
  niqe_por_imagen_6.csv                 NIQE del cribado inicial (todas las candidatas)
  resumen_{niqe,brisque,ma,musiq,pi}_6.csv
  wilcoxon_metricas_ciegas_6.csv        contrastes pareados
  rostros_6.csv                         rostros detectados / solape con máscara
  figuras/*.png                         cualitativa, boxplots, detalle de máscara
```
