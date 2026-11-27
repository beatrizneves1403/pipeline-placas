# Pipeline de segmentação e OCR de placas veiculares

Projeto didático para localizar placas nas imagens do dataset e reconhecer os caracteres com Tesseract OCR e EasyOCR. A segmentação usa o classificador Haar Cascade fornecido, com filtro de proporção e remoção de caixas sobrepostas. O recorte dos caracteres é ampliado e binarizado por Otsu antes do reconhecimento.

## Estrutura

- `dataset/imagens/`: imagens de entrada (adicione aqui os arquivos do dataset).
- `modelos/haarcascade_russian_plate_number.xml`: classificador Haar incluído.
- `src/deteccao.py`: detecção, extração da região e pré-processamento.
- `src/tesseract_ocr.py`: pipeline completo com Tesseract.
- `src/easyocr_ocr.py`: pipeline completo com EasyOCR.
- `placas_extraidas/`: recortes processados por abordagem.
- `resultados/`: um `.txt` por imagem e por abordagem.

## Instalação

Requer Python 3.10 ou superior. Instale as dependências com:

```bash
pip install -r requirements.txt
```

Instale também o executável do Tesseract OCR no sistema. Se ele não estiver no `PATH`, informe seu caminho antes da execução:

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

O EasyOCR baixa os modelos na primeira execução. A execução funciona em CPU, sem GPU.

## Execução

Na raiz do repositório, com as imagens em `dataset/imagens/`:

```bash
python src/tesseract_ocr.py
python src/easyocr_ocr.py
```

Os textos são salvos em `resultados/tesseract/` e `resultados/easyocr/`. Os recortes usados pelo OCR são salvos em `placas_extraidas/tesseract/` e `placas_extraidas/easyocr/`. Cada `.txt` registra os resultados encontrados na imagem com o mesmo nome.

## Etapas de segmentação

1. Converter a imagem para tons de cinza.
2. Executar `detectMultiScale` com o Haar Cascade.
3. Manter caixas horizontais com proporções compatíveis com placas e remover duplicatas sobrepostas.
4. Extrair a região candidata com pequena margem.
5. Remover parte da faixa superior e bordas, ampliar, suavizar e aplicar limiarização de Otsu.
6. Enviar a imagem preparada ao OCR e salvar recorte e texto.

A detecção Haar pode deixar de localizar placas pequenas, inclinadas ou muito diferentes do conjunto de treinamento. OCR é uma estimativa e pode confundir caracteres; confira os recortes e resultados antes de uso real.
