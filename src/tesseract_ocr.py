from pathlib import Path
import cv2
import pytesseract
from deteccao import carregar_detector, detectar_placas, obter_imagens, preparar_placa

RAIZ = Path(__file__).resolve().parent.parent
OUT = RAIZ / "resultados" / "tesseract"
RECORTES = RAIZ / "placas_extraidas" / "tesseract"


def executar():
    OUT.mkdir(parents=True, exist_ok=True)
    RECORTES.mkdir(parents=True, exist_ok=True)
    # Permite configurar TESSERACT_CMD; caso contrário, usa o PATH do sistema.
    caminho = __import__("os").environ.get("TESSERACT_CMD")
    if caminho:
        pytesseract.pytesseract.tesseract_cmd = caminho
    detector = carregar_detector()
    for arquivo in obter_imagens():
        imagem = cv2.imread(str(arquivo))
        textos = []
        if imagem is not None:
            for i, placa in enumerate(detectar_placas(imagem, detector), 1):
                processada = preparar_placa(placa)
                nome = f"{arquivo.stem}_placa_{i}"
                cv2.imwrite(str(RECORTES / f"{nome}.png"), processada)
                cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                texto = pytesseract.image_to_string(processada, lang="eng", config=cfg).strip()
                textos.append(f"Placa {i}: {texto or 'Texto não reconhecido'}")
        if not textos:
            textos = ["Placa não detectada." if imagem is not None else "Não foi possível abrir a imagem."]
        (OUT / f"{arquivo.stem}.txt").write_text("\n".join(textos) + "\n", encoding="utf-8")
        print(f"Processado: {arquivo.name}")

if __name__ == "__main__":
    executar()
