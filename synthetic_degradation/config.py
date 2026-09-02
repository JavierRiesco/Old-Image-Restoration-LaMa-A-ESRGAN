"""Configuración de rangos para la degradación sintética.

Todos los parámetros que controlan realismo viven aquí, para poder ajustarlos
sin tocar el código de dibujo. Los `tuple[a, b]` son rangos cerrados [a, b]
de los que se muestrea aleatoriamente.
"""

from dataclasses import dataclass


@dataclass
class DamageConfig:
    # --- Daño local: arañazos / rayas ---
    scratch_count: tuple[int, int] = (1, 4)          # nº de arañazos
    scratch_thickness: tuple[int, int] = (1, 3)      # grosor en px
    scratch_curved_prob: float = 0.4                 # prob. de curva Bézier
    scratch_dark_prob: float = 0.5                   # prob. oscuro vs claro
    scratch_alpha: tuple[float, float] = (0.6, 0.95) # opacidad de mezcla

    # --- Daño local: grietas / pliegues ---
    crack_count: tuple[int, int] = (0, 2)
    crack_thickness: tuple[int, int] = (2, 5)
    crack_segments: tuple[int, int] = (4, 10)        # nº de segmentos del random walk
    crack_step: tuple[int, int] = (10, 40)           # px por segmento
    crack_branch_prob: float = 0.3                   # prob. de ramificación
    crack_angle_delta: tuple[float, float] = (-0.6, 0.6)  # giro por segmento (rad)
    crack_alpha: tuple[float, float] = (0.7, 0.95)

    # --- Daño local: manchas / desconchados ---
    stain_count: tuple[int, int] = (0, 3)
    stain_radius_frac: tuple[float, float] = (0.03, 0.12)  # radio rel. a min(H, W)
    stain_light_prob: float = 0.6                    # prob. claro (pérdida emulsión)
    stain_alpha: tuple[float, float] = (0.4, 0.8)

    # --- Daño local: roturas de borde / esquina ---
    tear_count: tuple[int, int] = (0, 1)
    tear_size_frac: tuple[float, float] = (0.05, 0.18)  # tamaño rel. al borde
    tear_dark_prob: float = 0.5

    # --- Post-proceso de máscara ---
    mask_dilation_px: int = 1                         # margen de seguridad (px)

    # --- Efectos globales (probabilidad de aplicarse, rango de intensidad) ---
    sepia_prob: float = 0.8
    sepia_strength: tuple[float, float] = (0.3, 0.9)
    fade_prob: float = 0.7
    fade_strength: tuple[float, float] = (0.1, 0.4)
    blur_prob: float = 0.4
    blur_sigma: tuple[float, float] = (0.5, 1.5)
    grain_prob: float = 0.6
    grain_std: tuple[float, float] = (3.0, 12.0)
    vignette_prob: float = 0.4
    vignette_strength: tuple[float, float] = (0.2, 0.5)

    # --- Filtro de dataset ---
    min_mask_coverage: float = 0.01                  # descartar muestras con <1% de daño

    @classmethod
    def vintage_base(cls) -> "DamageConfig":
        """Preset para fuentes que YA son fotografías antiguas.

        Desactiva todos los efectos globales de época (sepia/fade/blur/grano/
        viñeta) para no provocar doble envejecimiento: el tono de época ya está
        en la imagen fuente. Con las probabilidades globales a 0, `apply_global`
        devuelve la imagen sin cambios, de modo que GT == la imagen vintage
        limpia y solo se añade daño local.
        """
        return cls(
            sepia_prob=0.0,
            fade_prob=0.0,
            blur_prob=0.0,
            grain_prob=0.0,
            vignette_prob=0.0,
        )

    @classmethod
    def tone_only(cls) -> "DamageConfig":
        """Preset solo-tono: aplica viraje sepia/B&N SIN desenfoque, grano,
        fade ni viñeta, y sin daño local. Para tonear imágenes HR nítidas a
        'vintage nítido' (GT de super-resolución). El sepia es un remapeo de
        color por píxel: preserva la estructura/nitidez.
        """
        return cls(
            sepia_prob=1.0,
            sepia_strength=(0.7, 1.0),
            fade_prob=0.0,
            blur_prob=0.0,
            grain_prob=0.0,
            vignette_prob=0.0,
            scratch_count=(0, 0),
            crack_count=(0, 0),
            stain_count=(0, 0),
            tear_count=(0, 0),
        )
