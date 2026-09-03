"""
dataset_espectral.py — Fase 4b, versión blur
Pipeline: H_espectral (filtro espectral del objetivo histórico) +
          GaussianBlur HR pre-submuestreo (sigma_blur) +
          grano correlacionado acromático + escaneo JPEG.

Hereda de torch.utils.data.Dataset directamente para evitar la dependencia
de RealESRGANDataset.__init__, que exige claves de degradación en GPU
incompatibles con high_order_degradation: false.
"""
import json, os, random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


# ── Carga del objetivo espectral ─────────────────────────────────────────────
def _cargar_psd(ruta_json: Path):
    candidatos = [
        ruta_json.parent / 'psd_objetivo.npz',
        ruta_json.parent.parent.parent / 'Fase4a' / 'resultados' / 'psd_objetivo.npz',
    ]
    for c in candidatos:
        if c.exists():
            return np.load(c)
    raise FileNotFoundError(
        f'psd_objetivo.npz no encontrado junto a {ruta_json} '
        f'ni en Fase4a/resultados/'
    )


def _construir_h_needed(psd_npz):
    freq  = psd_npz['freq'].astype(np.float64)
    hueco = psd_npz['hueco'].astype(np.float64)
    H = np.clip(10 ** (hueco / 2.0), 0.0, 1.0)
    return freq, H


# ── Filtro espectral ──────────────────────────────────────────────────────────
_MALLA_CACHE: dict = {}

def _malla_rfft(h: int, w: int):
    key = (h, w)
    if key not in _MALLA_CACHE:
        fy = np.fft.fftfreq(h).astype(np.float32)[:, None]
        fx = np.fft.rfftfreq(w).astype(np.float32)[None, :]
        _MALLA_CACHE[key] = np.sqrt(fy**2 + fx**2)
    return _MALLA_CACHE[key]


def _h_espectral(h, w, escala, freq_lq, H_needed, beta):
    rho_hr = _malla_rfft(h, w)
    rho_lq = rho_hr * escala
    H_beta = H_needed ** beta
    mask = np.interp(rho_lq.ravel(), freq_lq, H_beta, left=1.0, right=0.0)
    return mask.reshape(rho_lq.shape).astype(np.float32)


def _aplicar_h_espectral(img, escala, freq_lq, H_needed, beta):
    h, w = img.shape[:2]
    H = _h_espectral(h, w, escala, freq_lq, H_needed, beta)
    if img.ndim == 2:
        return np.fft.irfft2(np.fft.rfft2(img) * H, s=(h, w)).astype(np.float32)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[..., c] = np.fft.irfft2(np.fft.rfft2(img[..., c]) * H, s=(h, w))
    return out


# ── Grano correlacionado acromático ──────────────────────────────────────────
def _aplicar_grano(img, sigma, alpha, rng):
    if sigma <= 0:
        return img
    h, w = img.shape[:2]
    ruido = rng.standard_normal((h, w)).astype(np.float32)
    F = np.fft.rfft2(ruido)
    fy = np.fft.fftfreq(h).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(w).astype(np.float32)[None, :]
    rho = np.sqrt(fy**2 + fx**2)
    filtro = np.where(rho > 0, rho ** (-alpha / 2.0), 0.0).astype(np.float32)
    grano = np.fft.irfft2(F * filtro, s=(h, w)).astype(np.float32)
    grano = grano / (grano.std() + 1e-8) * sigma
    if img.ndim == 3:
        grano = grano[..., None]
    return img + grano


# ── Escaneo ───────────────────────────────────────────────────────────────────
def _escaneo(img, escala, sigma_blur, sigma_ruido, jpeg_q, rng):
    if sigma_blur > 0.0:
        ksize = int(sigma_blur * 6) | 1
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma_blur)
    h, w = img.shape[:2]
    lq = cv2.resize(img, (w // escala, h // escala), interpolation=cv2.INTER_AREA)
    if sigma_ruido > 0:
        n = rng.normal(0, sigma_ruido, lq.shape[:2]).astype(np.float32)
        lq = lq + (n[..., None] if lq.ndim == 3 else n)
    lq = np.clip(lq, 0, 1)
    if jpeg_q < 100:
        ok, enc = cv2.imencode(
            '.jpg', (lq * 255).astype(np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)]
        )
        if ok:
            flag = cv2.IMREAD_COLOR if lq.ndim == 3 else cv2.IMREAD_GRAYSCALE
            lq = cv2.imdecode(enc, flag).astype(np.float32) / 255.0
    return np.clip(lq, 0, 1)


# ── Dataset ───────────────────────────────────────────────────────────────────
@DATASET_REGISTRY.register()
class RealESRGANDatasetEspectral(Dataset):
    """Dataset de super-resolución con degradación espectral calibrada.

    Hereda de torch.utils.data.Dataset para evitar que RealESRGANDataset.__init__
    exija claves de degradación en GPU (blur_kernel_size, etc.) incompatibles
    con high_order_degradation: false.

    Claves requeridas en el YAML:
        dataroot_gt     (str)  : carpeta con las imágenes HR
        gt_size         (int)  : tamaño del parche de entrenamiento (ej. 128)
        ruta_parametros (str)  : ruta al JSON con los rangos calibrados
        io_backend.type (str)  : disk

    Claves opcionales:
        use_hflip (bool), use_rot (bool), meta_info (str)
    """

    def __init__(self, opt):
        super().__init__()
        self.opt      = opt
        self.gt_size  = int(opt['gt_size'])
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot   = opt.get('use_rot',   True)

        self.file_client     = None
        self.io_backend_opt  = dict(opt.get('io_backend', {'type': 'disk'}))

        # Lista de imágenes
        dataroot  = opt['dataroot_gt']
        meta_info = opt.get('meta_info')
        if meta_info and Path(meta_info).exists():
            self.paths = [
                str(Path(dataroot) / l.strip())
                for l in Path(meta_info).read_text().splitlines() if l.strip()
            ]
        else:
            self.paths = sorted(str(p) for p in Path(dataroot).rglob('*.png'))
        assert self.paths, f'Sin imágenes en {dataroot}'

        # Degradación espectral
        ruta = Path(opt['ruta_parametros'])
        self._p        = json.loads(ruta.read_text())
        psd            = _cargar_psd(ruta)
        self._freq_lq, self._H_needed = _construir_h_needed(psd)
        self._escala   = int(self._p.get('escala', 4))
        self._rng      = np.random.default_rng()

    def __len__(self):
        return len(self.paths)

    def _muestrear(self, clave, por_defecto):
        r = self._p.get(clave)
        return float(self._rng.uniform(r[0], r[1])) if r else float(por_defecto)

    def _muestrear_int(self, clave, por_defecto):
        r = self._p.get(clave)
        return int(round(self._rng.uniform(r[0], r[1]))) if r else int(por_defecto)

    def __getitem__(self, index):
        if self.file_client is None:
            backend_opt = dict(self.io_backend_opt)
            backend_type = backend_opt.pop('type', 'disk')
            self.file_client = FileClient(backend_type, **backend_opt)

        path      = self.paths[index % len(self.paths)]
        img_bytes = self.file_client.get(path)
        gt_np     = imfrombytes(img_bytes, float32=True)   # HWC BGR float32 [0,1]

        # Recorte aleatorio
        h, w = gt_np.shape[:2]
        if h < self.gt_size or w < self.gt_size:
            gt_np = cv2.resize(gt_np, (self.gt_size, self.gt_size))
            h, w  = self.gt_size, self.gt_size
        top  = random.randint(0, h - self.gt_size)
        left = random.randint(0, w - self.gt_size)
        gt_np = gt_np[top:top + self.gt_size, left:left + self.gt_size]

        # Aumentación geométrica
        if self.use_hflip and random.random() < 0.5:
            gt_np = np.ascontiguousarray(gt_np[:, ::-1])
        if self.use_rot:
            k = random.randint(0, 3)
            if k:
                gt_np = np.ascontiguousarray(np.rot90(gt_np, k))

        # Pipeline de degradación espectral
        beta       = self._muestrear('beta_rango',          1.0)
        sigma_blur = self._muestrear('sigma_blur_rango',    0.0)
        sig_grano  = self._muestrear('sigma_grano_rango',   0.035)
        alpha_gran = self._muestrear('alpha_grano_rango',   1.0)
        sig_esc    = self._muestrear('sigma_escaneo_rango', 0.0)
        jpeg_q     = self._muestrear_int('jpeg_rango',      95)

        lq_np = _aplicar_h_espectral(
            gt_np, self._escala, self._freq_lq, self._H_needed, beta
        )
        lq_np = np.clip(lq_np, 0, 1)
        lq_np = _aplicar_grano(lq_np, sig_grano, alpha_gran, self._rng)
        lq_np = np.clip(lq_np, 0, 1)
        lq_np = _escaneo(lq_np, self._escala, sigma_blur, sig_esc, jpeg_q, self._rng)

        # HWC BGR → tensor CHW RGB float32
        gt_t = img2tensor(gt_np,  bgr2rgb=True, float32=True)
        lq_t = img2tensor(lq_np,  bgr2rgb=True, float32=True)

        return {'gt': gt_t, 'lq': lq_t, 'gt_path': path}
