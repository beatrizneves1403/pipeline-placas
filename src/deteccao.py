from pathlib import Path
import cv2

RAIZ = Path(__file__).resolve().parent.parent
PASTA_IMAGENS = RAIZ / "dataset" / "imagens"
ARQUIVO_CASCADE = RAIZ / "modelos" / "haarcascade_russian_plate_number.xml"
EXTENSOES = {".jpg", ".jpeg", ".png"}


def carregar_detector():
    """Carrega o classificador Haar incluído em modelos/."""
    detector = cv2.CascadeClassifier(str(ARQUIVO_CASCADE))
    if detector.empty():
        raise FileNotFoundError(f"Classificador não encontrado: {ARQUIVO_CASCADE}")
    return detector


def obter_imagens():
    """Lista imagens da pasta dataset/imagens, em ordem previsível."""
    if not PASTA_IMAGENS.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {PASTA_IMAGENS}")
    return sorted(p for p in PASTA_IMAGENS.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTENSOES)


def detectar_placas(imagem, detector):
    """Detecta candidatos Haar, filtra proporções de placas e remove duplicatas."""
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    altura, largura = cinza.shape
    candidatos = detector.detectMultiScale(
        cinza, scaleFactor=1.05, minNeighbors=3, minSize=(50, 18)
    )
    validos = []
    for x, y, w, h in candidatos:
        proporcao = w / max(h, 1)
        # Placas brasileiras são horizontais; o filtro elimina muitos falsos positivos.
        if 2.0 <= proporcao <= 6.5 and w * h >= largura * altura * 0.0005:
            validos.append((x, y, w, h))

    # Suprime caixas sobrepostas, mantendo primeiro as maiores.
    validos.sort(key=lambda caixa: caixa[2] * caixa[3], reverse=True)
    unicos = []
    for caixa in validos:
        x, y, w, h = caixa
        sobrepoe = False
        for ax, ay, aw, ah in unicos:
            ix, iy = max(x, ax), max(y, ay)
            ir, ib = min(x + w, ax + aw), min(y + h, ay + ah)
            inter = max(0, ir - ix) * max(0, ib - iy)
            uniao = w * h + aw * ah - inter
            if min(w * h, aw * ah) and inter / min(w * h, aw * ah) > 0.50:
                sobrepoe = True
                break
        if not sobrepoe:
            unicos.append(caixa)

    recortes = []
    for x, y, w, h in unicos:
        # Margem pequena mantém a borda dos caracteres sem incluir muita carroceria.
        mx, my = int(w * 0.025), int(h * 0.08)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(largura, x + w + mx), min(altura, y + h + my)
        roi = imagem[y1:y2, x1:x2]
        if roi.size:
            recortes.append(roi)
    # Mantém o candidato maior para reduzir falsos positivos neste conjunto.
    return recortes[:1]


def preparar_placa(placa):
    """Recorta caracteres, amplia e binariza para reduzir ruído antes do OCR."""
    h, w = placa.shape[:2]
    # Remove parte superior azul, bordas e área 'BR'; preserva os sete caracteres.
    y1, y2 = int(h * 0.28), int(h * 0.88)
    x1, x2 = int(w * 0.12), int(w * 0.96)
    regiao = placa[y1:y2, x1:x2]
    if regiao.size == 0:
        regiao = placa
    cinza = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    ampliada = cv2.resize(cinza, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    suavizada = cv2.GaussianBlur(ampliada, (3, 3), 0)
    _, binaria = cv2.threshold(suavizada, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(binaria, 18, 18, 18, 18,
                              cv2.BORDER_CONSTANT, value=255)






