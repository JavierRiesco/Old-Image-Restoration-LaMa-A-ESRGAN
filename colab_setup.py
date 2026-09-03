"""Plumbing del entorno Google Colab para los notebooks del proyecto.

Este modulo concentra el codigo de preparacion de entorno que hasta ahora se
copiaba y pegaba en cada notebook: parches de dependencias, clonado del repo y
descarga de datasets/artefactos.

Nada se ejecuta al importar. Todas las funciones deben invocarse
explicitamente. Fuera de Colab las funciones que tocan red o pip
(``aplicar_parches``, la rama Colab de ``preparar_repo``, ``descargar_datos``)
no deben llamarse; en cambio las constantes (``PINS``, ``_FUENTES``, ...) y los
helpers puros (``_url_release``, ``_destino``, ``_en_colab``, ...) son seguros en
cualquier entorno.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PINS = ("numpy==2.0.2", "scipy==1.14.1", "scikit-image==0.24.0")

_FUENTES = {
    "kaggle:old-photos": ("kaggle", "marcinrutecki/old-photos"),
    "kaggle:vintage-degraded": ("kaggle", "shrutimandaokar2301/vintage-degraded-image-synthetic-real"),
    "kaggle:div2k": ("kaggle", "joe1995/div2k-dataset"),
    "kaggle:flickr2k": ("kaggle", "daehoyang/flickr2k"),
    "release:lama_synthetic": ("release", "lama_synthetic.zip"),
    "release:lama_ft": ("release", "lama_finetuned.zip"),
    "release:aesrgan_ft": ("release", "aesrgan_finetuned.zip"),
    "release:vintage_degraded": ("release", "vintage_degraded.zip"),
}
_RELEASE_BASE = "https://github.com/JavierRiesco/Old-Image-Restoration-LaMa-A-ESRGAN/releases/download/v1.0"
_REPO_URL = "https://github.com/JavierRiesco/Old-Image-Restoration-LaMa-A-ESRGAN.git"


def _en_colab() -> bool:
    """True si estamos ejecutando dentro de Google Colab."""
    return "google.colab" in sys.modules or Path("/content").is_dir()


def aplicar_parches() -> None:
    """Fija versiones de numpy/scipy/scikit-image y aplica parches de compatibilidad.

    Solo debe llamarse dentro de Colab: reinstala paquetes con pip.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", *PINS],
        check=True,
    )
    destino_pins = Path("/content/numpy_pins.txt") if _en_colab() else Path("numpy_pins.txt")
    destino_pins.write_text("\n".join(PINS) + "\n")
    os.environ["PIP_CONSTRAINT"] = str(destino_pins)

    _parche_pillow_mode()
    _parche_basicsr_functional_tensor()
    _parche_pil_typing_ink()


def _parche_pillow_mode() -> None:
    """Hace escribible el atributo ``mode`` de las imagenes JPEG de Pillow."""
    import PIL.JpegImagePlugin

    for cls in PIL.JpegImagePlugin.JpegImageFile.__mro__:
        if "mode" in cls.__dict__:
            prop = cls.__dict__["mode"]
            if isinstance(prop, property) and prop.fset is None:
                cls.mode = prop.setter(
                    lambda self, v: object.__setattr__(self, "_mode", v)
                )
            break


def _parche_basicsr_functional_tensor() -> None:
    """Corrige el import roto de torchvision en basicsr.data.degradations."""
    import site

    viejo = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    nuevo = "from torchvision.transforms.functional import rgb_to_grayscale"
    for directorio in site.getsitepackages() + [site.getusersitepackages()]:
        ruta = Path(directorio) / "basicsr" / "data" / "degradations.py"
        if ruta.exists():
            contenido = ruta.read_text()
            if viejo in contenido:
                ruta.write_text(contenido.replace(viejo, nuevo))


def _parche_pil_typing_ink() -> None:
    """Restaura ``PIL._typing._Ink`` cuando falta (incompatibilidad basicsr/Pillow)."""
    from typing import Union

    import PIL._typing

    if not hasattr(PIL._typing, "_Ink"):
        PIL._typing._Ink = Union[int, tuple]


def _url_release(nombre_zip: str) -> str:
    return f"{_RELEASE_BASE}/{nombre_zip}"


def _destino(nombre: str, root: Path) -> Path:
    return root / "_data" / nombre.split(":", 1)[1]


def descargar_datos(*que: str, root: Path) -> dict[str, Path]:
    """Descarga datasets de Kaggle o artefactos de Releases a ``root/_data/``.

    Toca red (kagglehub / urllib). No invocar fuera de Colab.
    """
    import shutil
    import urllib.request
    import zipfile

    resultado: dict[str, Path] = {}
    for nombre in que:
        tipo, ref = _FUENTES[nombre]
        destino = _destino(nombre, root)

        if destino.is_dir() and any(destino.iterdir()):
            resultado[nombre] = destino
            continue

        destino.mkdir(parents=True, exist_ok=True)

        if tipo == "kaggle":
            import kagglehub

            origen = Path(kagglehub.dataset_download(ref))
            for archivo in origen.rglob("*"):
                if archivo.is_file():
                    relativo = archivo.relative_to(origen)
                    salida = destino / relativo
                    salida.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(archivo, salida)
        elif tipo == "release":
            zip_tmp = destino.parent / f"_{destino.name}.zip"
            urllib.request.urlretrieve(_url_release(ref), zip_tmp)
            with zipfile.ZipFile(zip_tmp) as zf:
                zf.extractall(destino)
            zip_tmp.unlink()

        resultado[nombre] = destino

    return resultado


def preparar_repo(rama: str = "main") -> Path:
    """Devuelve la raiz del repo, clonandolo e instalandolo si estamos en Colab."""
    if not _en_colab():
        actual = Path.cwd()
        for candidato in [actual, *actual.parents]:
            if (candidato / "colab_setup.py").is_file():
                return candidato
        raise RuntimeError("No se encontro colab_setup.py subiendo desde el cwd")

    raiz = Path("/content/Old-Image-Restoration-LaMa-A-ESRGAN")
    if not raiz.exists():
        subprocess.run(
            ["git", "clone", "--branch", rama, "--depth", "1", _REPO_URL, str(raiz)],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(raiz)],
        check=True,
    )
    if str(raiz) not in sys.path:
        sys.path.insert(0, str(raiz))
    return raiz


def montar_drive_opcional() -> Path | None:
    """Monta Google Drive en Colab y devuelve la carpeta TFM; fuera de Colab None."""
    if not _en_colab():
        return None
    from google.colab import drive

    drive.mount("/content/drive")
    return Path("/content/drive/MyDrive/TFM")
