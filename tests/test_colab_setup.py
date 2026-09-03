"""Tests de las ramas puras de colab_setup (sin red ni pip)."""
import colab_setup
import pytest


def test_modulo_importa_sin_efectos():
    assert hasattr(colab_setup, "aplicar_parches")
    assert hasattr(colab_setup, "preparar_repo")
    assert hasattr(colab_setup, "descargar_datos")
    assert hasattr(colab_setup, "montar_drive_opcional")


def test_pins_declarados():
    assert colab_setup.PINS == ("numpy==2.0.2", "scipy==1.14.1", "scikit-image==0.24.0")


def test_descargar_datos_rechaza_nombre_desconocido(tmp_path):
    with pytest.raises(KeyError):
        colab_setup.descargar_datos("kaggle:no-existe", root=tmp_path)


def test_url_de_release_se_construye_bien():
    assert colab_setup._url_release("lama_synthetic.zip") == (
        "https://github.com/JavierRiesco/Old-Image-Restoration-LaMa-A-ESRGAN"
        "/releases/download/v1.0/lama_synthetic.zip"
    )


def test_destino_por_nombre(tmp_path):
    assert colab_setup._destino("release:lama_ft", tmp_path) == tmp_path / "_data" / "lama_ft"
    assert colab_setup._destino("kaggle:div2k", tmp_path) == tmp_path / "_data" / "div2k"


def test_preparar_repo_devuelve_raiz_fuera_de_colab(monkeypatch, tmp_path):
    monkeypatch.setattr(colab_setup, "_en_colab", lambda: False)
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    (tmp_path / "colab_setup.py").write_text("")
    monkeypatch.chdir(sub)
    assert colab_setup.preparar_repo() == tmp_path


def test_montar_drive_none_fuera_de_colab(monkeypatch):
    monkeypatch.setattr(colab_setup, "_en_colab", lambda: False)
    assert colab_setup.montar_drive_opcional() is None


def test_en_colab_falso_en_entorno_de_test():
    assert colab_setup._en_colab() is False


def test_fuentes_tienen_tipo_valido():
    for nombre, (tipo, ref) in colab_setup._FUENTES.items():
        assert tipo in ("kaggle", "release")
        assert isinstance(ref, str) and ref


def test_url_release_usa_base_de_v1():
    assert colab_setup._url_release("x.zip").startswith(colab_setup._RELEASE_BASE + "/")


def test_descargar_datos_salta_si_destino_no_vacio(tmp_path):
    destino = colab_setup._destino("release:lama_ft", tmp_path)
    destino.mkdir(parents=True)
    (destino / "ya_esta.txt").write_text("contenido")
    res = colab_setup.descargar_datos("release:lama_ft", root=tmp_path)
    assert res == {"release:lama_ft": destino}
