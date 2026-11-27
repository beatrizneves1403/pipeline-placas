from pathlib import Path
import cv2
import easyocr
from deteccao import carregar_detector, detectar_placas, obter_imagens, preparar_placa

RAIZ = Path(__file__).resolve().parent.parent
OUT = RAIZ / "resultados" / "easyocr"
RECORTES = RAIZ / "placas_extraidas" / "easyocr"


def executar():
    OUT.mkdir(parents=True, exist_ok=True)
    RECORTES.mkdir(parents=True, exist_ok=True)
    detector = carregar_detector()
    imagens = obter_imagens()
    if not imagens:
        print("Nenhuma imagem encontrada em dataset/imagens.")
        return
    # O primeiro uso pode baixar os modelos oficiais do EasyOCR.
    leitor = easyocr.Reader(["en"], gpu=False)
    for arquivo in imagens:
        imagem = cv2.imread(str(arquivo))
        textos = []
        if imagem is not None:
            for i, placa in enumerate(detectar_placas(imagem, detector), 1):
                processada = preparar_placa(placa)
                nome = f"{arquivo.stem}_placa_{i}"
                cv2.imwrite(str(RECORTES / f"{nome}.png"), processada)
                partes = leitor.readtext(processada, detail=0,
                                         allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                texto = "".join(partes).strip()
                textos.append(f"Placa {i}: {texto or 'Texto não reconhecido'}")
        if not textos:
            textos = ["Placa não detectada." if imagem is not None else "Não foi possível abrir a imagem."]
        (OUT / f"{arquivo.stem}.txt").write_text("\n".join(textos) + "\n", encoding="utf-8")
        print(f"Processado: {arquivo.name}")

if __name__ == "__main__":
    executar()
